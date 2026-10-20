"""NLE log-likelihood of an observation (and of held-out null events) at EVERY mock's parameters.

WHAT THIS IS FOR
----------------
Posterior predictive checks (PPC) of the real-data NLE posteriors
(`.claude/runs/eval-and-viz/vdq-posterior-predictive/plan.md`). We cannot run Gower Street sims at
new cosmologies, so the posterior predictive distribution of a statistic T is built by
IMPORTANCE-WEIGHTING the existing mock suite. Every mock block b of sim i (one (sim, outer, rot)
triple; its inner shape-noise files share the block's parameters) gets

    log w_ib = log L(t_obs | theta_ib) + log pi(theta_ib) - log q(theta_ib)

where L is the trained ensemble NLE likelihood, pi the analysis prior and q the density the suite
was generated from. This module computes the expensive term, log L(t_e | theta_b), for the real
observation AND for every held-out null event e (the test-lock mocks, used to calibrate the PPC
p-value), at every block of a "theta store". The priors are evaluated here too (they are cheap and
must match the sampler's), but the weights themselves are formed offline.

THE INVARIANT
-------------
`log L` is EXACTLY the potential the reported posterior was sampled with:
  * the observation goes through `resolve_nle_pipeline` (training scalers injected, Stage-A
    whitener loaded, per-member frames when persisted -- the nle-external path, not a copy);
  * the likelihood is `build_grouped_ensemble_likelihood(members)` (logmeanexp over members, the
    sampler's fast path) evaluated on the ensemble's own `test_dataloader` rows, which is what
    `EnsembleLikelihoodNDELightningModule.generate_samples` feeds the MCMC;
  * theta is put in the sampler's scaled space with the pipeline's own cosmo scaler.
A self-check proves the theta scaling: for every null event (a mock with a truth vector) the
loader's scaled truth must equal this module's scaled table row for the same file.

OUTPUT
------
`{base_path}/checkpoints/<exp>/external/ppc_<tag>/ppc_loglik_<theta_store>_<match>.npz` with
  event_files [E]           event basenames; row 0 is the observation when one is given
  event_is_obs [E]
  event_theta_scaled [E,D]  loader truth (NaN for the observation)
  event_ok [E]              truth check passed (always True for the observation)
  loglik [E,B] float32      ensemble log L(t_e | theta_b)
  loglik_obs_members [M,B]  per-member log L for the observation (diagnostic only)
  block_sim, block_out, block_rot [B]; block_theta_raw [B,D]; block_theta_scaled [B,D]
  file_names [F], file_block [F]  every theta-store file and its block index
  extra_names [K], block_extras [B,K]  every other scalar in cosmo_dict (b_g_bin*, ...)
  logprior_<mode> [B]       analysis / gower prior log-density in scaled space (when buildable)
plus a provenance json.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

_FNAME_RE = re.compile(r"output_(\d+)_out(\d+)_rot(\d+)_(\d+)\.h5$")


# --------------------------------------------------------------------------------------------
# theta table: one row per (sim, outer, rot) block of a mock store
# --------------------------------------------------------------------------------------------
def parse_mock_name(path: str):
    """`output_<sim>_out<o>_rot<r>_<k>.h5` -> (sim, o, r, k), or None for any other name."""
    m = _FNAME_RE.search(os.path.basename(str(path)))
    return None if m is None else tuple(int(g) for g in m.groups())


def _read_cosmo_dict(path: str) -> Optional[Dict[str, float]]:
    """Every SCALAR numeric entry of the file's cosmo_dict (strings such as `ia_model` are dropped).
    None when the file cannot be opened (truncated / corrupt) -- such files are excluded."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            g = f["cosmo_dict"]
            out = {}
            for k in g.keys():
                v = np.asarray(g[k][()])
                if v.ndim == 0 and np.issubdtype(v.dtype, np.number):
                    out[str(k)] = float(v)
            return out
    except (OSError, KeyError):
        return None


def build_theta_table(paths: Sequence[str], param_names: Sequence[str], *, n_threads: int = 16,
                      atol: float = 1e-6) -> Dict[str, object]:
    """Read every file's cosmo_dict, group files into (sim, out, rot) blocks, return the block table.

    The inner files of a block must share every inferred parameter (the simulator draws the
    nuisances per block); any disagreement beyond `atol` is counted and reported, never hidden.
    """
    paths = sorted(str(p) for p in paths)
    keyed = [(p, parse_mock_name(p)) for p in paths]
    keyed = [(p, k) for p, k in keyed if k is not None]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(n_threads))) as ex:
        dicts = list(ex.map(_read_cosmo_dict, [p for p, _ in keyed]))
    print(f"[ppc-loglik] read cosmo_dict of {len(keyed)} files in {time.time() - t0:.0f}s", flush=True)

    param_names = list(param_names)
    blocks: Dict[tuple, int] = {}
    b_rows: List[Dict[str, float]] = []
    file_names, file_block, n_bad, n_missing_param, n_inconsistent = [], [], 0, 0, 0
    for (p, (sim, o, r, k)), d in zip(keyed, dicts):
        if d is None:
            n_bad += 1
            continue
        if any(n not in d for n in param_names):
            n_missing_param += 1
            continue
        key = (sim, o, r)
        if key not in blocks:
            blocks[key] = len(b_rows)
            b_rows.append(d)
        else:
            ref = b_rows[blocks[key]]
            if max(abs(ref[n] - d[n]) for n in param_names) > atol:
                n_inconsistent += 1
        file_names.append(os.path.basename(p))
        file_block.append(blocks[key])

    keys = sorted(blocks, key=lambda kk: blocks[kk])
    extra_names = sorted({k for d in b_rows for k in d} - set(param_names))
    table = {
        "block_sim": np.array([k[0] for k in keys], dtype=np.int32),
        "block_out": np.array([k[1] for k in keys], dtype=np.int16),
        "block_rot": np.array([k[2] for k in keys], dtype=np.int16),
        "block_theta_raw": np.array([[d[n] for n in param_names] for d in b_rows], dtype=np.float64),
        "extra_names": np.array(extra_names),
        "block_extras": np.array([[d.get(n, np.nan) for n in extra_names] for d in b_rows],
                                 dtype=np.float64),
        "file_names": np.array(file_names),
        "file_block": np.array(file_block, dtype=np.int32),
        "param_names": np.array(param_names),
    }
    stats = {"n_files_seen": len(keyed), "n_files_kept": len(file_names), "n_blocks": len(keys),
             "n_cosmologies": int(len(set(table["block_sim"].tolist()))), "n_unreadable": n_bad,
             "n_missing_param": n_missing_param, "n_inner_inconsistent": n_inconsistent}
    print(f"[ppc-loglik] theta table: {stats}", flush=True)
    return {"table": table, "stats": stats, "kept_paths": [p for (p, _), d in zip(keyed, dicts)
                                                          if d is not None and all(n in d for n in param_names)]}


def null_event_paths(kept_paths: Sequence[str], test_sim_ids: Sequence[int], inner: int = 0) -> List[str]:
    """The Stage-B in-distribution test events of a store: every block of a test-lock sim, inner
    realisation `inner` (the eval's `test_shape_noise_idx=[0]`). Sorted, so the order is a pure
    function of the store."""
    ids = set(int(i) for i in test_sim_ids)
    out = []
    for p in kept_paths:
        k = parse_mock_name(p)
        if k is not None and k[0] in ids and k[3] == int(inner):
            out.append(str(p))
    return sorted(out)


def load_lock_ids(lock_path) -> List[int]:
    with open(lock_path) as fh:
        d = json.load(fh)
    return list(d["sim_ids"] if isinstance(d, dict) else d)


# --------------------------------------------------------------------------------------------
# the likelihood matrix
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def loglik_matrix(loglik_fn, X: torch.Tensor, Theta: torch.Tensor, *, chunk: int = 16384,
                  log_every: int = 200) -> np.ndarray:
    """[E, B] matrix of `loglik_fn(x_e tiled, Theta_chunk)`; `loglik_fn(x[n,k], theta[n,D]) -> [n]`."""
    E, B = int(X.shape[0]), int(Theta.shape[0])
    out = np.empty((E, B), dtype=np.float32)
    t0 = time.time()
    for e in range(E):
        xe = X[e:e + 1]
        for s in range(0, B, chunk):
            th = Theta[s:s + chunk]
            out[e, s:s + th.shape[0]] = loglik_fn(xe.expand(th.shape[0], -1), th).reshape(-1).cpu().numpy()
        if log_every and (e + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[ppc-loglik] {e + 1}/{E} events ({el:.0f}s, ~{el / (e + 1) * (E - e - 1):.0f}s left)",
                  flush=True)
    return out


def _collect_loader(loader):
    xs, ths = [], []
    for x, th in loader:
        xs.append(x if torch.is_tensor(x) else torch.as_tensor(x))
        ths.append(th if torch.is_tensor(th) else torch.as_tensor(th))
    return torch.cat(xs, 0).float(), torch.cat(ths, 0).float()


def run_ppc_loglik(
    experiment: str,
    match_string: str,
    *,
    theta_store: Optional[str] = None,
    obs_store: Optional[str] = None,
    tag: str = "A2",
    null: str = "lock",
    max_null_events: Optional[int] = None,
    batch_size: int = 64,
    require_member_frames: bool = False,
    n_threads: Optional[int] = None,
    priors: Sequence[str] = ("kids_s8_analytic", "gower"),
    dry_run: bool = False,
) -> Dict[str, object]:
    from ..models.lightning.ensemble_flow import build_grouped_ensemble_likelihood
    from ..models.lightning.ensemble_nle import _EnsembleLikelihoodModel
    from .nle_external import (_jsonable, _paths_of, build_prior_for_mode, output_dir,
                               resolve_nle_pipeline, store_glob)
    from .utils import _config_preset_overrides

    if n_threads:
        torch.set_num_threads(int(n_threads))

    # Same config build as load_embedding_model_with_dataloader (defaults + experiment overrides).
    from config.experiments import experiments as _exps
    from ..embeddings.train import _extract_ncosmo_from_match, build_cfg_from_experiment_dict
    cfg0 = build_cfg_from_experiment_dict(experiment, _exps[experiment],
                                          n_cosmo=_extract_ncosmo_from_match(match_string))
    param_names = list(cfg0.cosmo_param_names)
    if theta_store is None:
        # Default: the experiment's OWN Gower store (the suite its posterior's prior-predictive is).
        pats = cfg0.data_patterns if isinstance(cfg0.data_patterns, str) else list(cfg0.data_patterns)[0]
        theta_store = os.path.basename(os.path.dirname(str(pats)))

    theta_paths = sorted(_glob.glob(store_glob(theta_store)))
    if not theta_paths:
        raise FileNotFoundError(f"theta store '{theta_store}' matched no files ({store_glob(theta_store)})")
    obs_paths = []
    if obs_store:
        obs_paths = sorted(_glob.glob(store_glob(obs_store)))
        if len(obs_paths) != 1:
            raise ValueError(f"obs store '{obs_store}' must hold exactly ONE file, found {len(obs_paths)}")

    # The parameter names come from the experiment config; resolve it cheaply first so the theta
    # table can be read before the (expensive) encoder pass.
    tt = build_theta_table(theta_paths, param_names, n_threads=n_threads or 16)
    table = tt["table"]

    null_paths: List[str] = []
    lock_path = getattr(cfg0, "fixed_test_sim_ids", None)
    if null == "lock":
        if not lock_path:
            raise ValueError(f"{experiment} has no fixed_test_sim_ids; cannot build the lock null")
        if not os.path.isabs(str(lock_path)) and not os.path.exists(str(lock_path)):
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            lock_path = os.path.join(repo, str(lock_path))
        null_paths = null_event_paths(tt["kept_paths"], load_lock_ids(lock_path))
        if max_null_events and len(null_paths) > int(max_null_events):
            # Deterministic thinning that keeps every cosmology: stride through the sorted list.
            stride = int(np.ceil(len(null_paths) / int(max_null_events)))
            null_paths = null_paths[::stride]
    paths = list(obs_paths) + list(null_paths)
    if not paths:
        raise ValueError("nothing to score: give --data-store (observation) and/or --ppc-null lock")
    print(f"[ppc-loglik] {experiment} {match_string}: obs={len(obs_paths)} null={len(null_paths)} "
          f"(lock={lock_path}) theta_store={theta_store} blocks={table['block_sim'].size}", flush=True)

    p = resolve_nle_pipeline(experiment, match_string, paths=paths, batch_size=batch_size,
                             require_member_frames=require_member_frames)
    if list(p.param_names) != param_names:
        raise RuntimeError(f"param order mismatch: config {param_names} vs pipeline {p.param_names}")
    out_dir = output_dir(p.config, f"ppc_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"ppc_loglik_{theta_store}_{p.match_string}"
    if dry_run:
        print(f"[ppc-loglik] DRY RUN -- would write {out_dir}/{stem}.npz", flush=True)
        return {"output_dir": out_dir, "dry_run": True}

    scored = [os.path.basename(s) for s in _paths_of(p, paths)]
    if scored != [os.path.basename(s) for s in paths]:
        raise RuntimeError("scored path order differs from the requested order")
    X, th_loader = _collect_loader(p.model.test_dataloader)          # member-0 rows == the sampler's
    if X.shape[0] != len(paths):
        raise RuntimeError(f"loader returned {X.shape[0]} rows for {len(paths)} paths")

    cosmo_scaler = p.scalers["cosmo"] if isinstance(p.scalers, dict) else None
    theta_scaled = np.asarray(cosmo_scaler.transform(table["block_theta_raw"]), dtype=np.float64)

    # SELF-CHECK: the loader's scaled truth of each null event == our scaled row for that file.
    name_to_block = dict(zip(table["file_names"].tolist(), table["file_block"].tolist()))
    is_obs = np.array([i < len(obs_paths) for i in range(len(paths))])
    ok = np.ones(len(paths), dtype=bool)
    max_dev = 0.0
    th_np = th_loader.numpy().astype(np.float64)
    for i, pth in enumerate(paths):
        if is_obs[i]:
            continue
        b = name_to_block.get(os.path.basename(pth))
        if b is None:
            ok[i] = False
            continue
        dev = float(np.nanmax(np.abs(th_np[i] - theta_scaled[b])))
        max_dev = max(max_dev, dev)
        ok[i] = dev < 1e-4
    print(f"[ppc-loglik] truth self-check: {int((~ok).sum())}/{int((~is_obs).sum())} null events "
          f"FAILED (max |dev| {max_dev:.2e})", flush=True)
    if (~is_obs).sum() and ok[~is_obs].mean() < 0.99:
        raise RuntimeError("more than 1% of null events fail the theta self-check -- the loader and "
                           "the theta table disagree (row misalignment or a different scaler)")

    members = list(p.model.members)
    for m in members:
        m.eval()
        m.to("cpu")
        if hasattr(m, "model") and hasattr(m.model, "embedding_net") and hasattr(m.model.embedding_net, "only_return_mu"):
            m.model.embedding_net.only_return_mu = True          # as build_posterior_object does
    grouped = build_grouped_ensemble_likelihood(members, reduction="logmeanexp")
    ens = grouped if grouped is not None else _EnsembleLikelihoodModel(members, reduction="logmeanexp")
    print(f"[ppc-loglik] ensemble likelihood: {'grouped (fast)' if grouped is not None else 'serial'} "
          f"over {len(members)} members", flush=True)

    Theta = torch.as_tensor(theta_scaled, dtype=torch.float32)
    t0 = time.time()
    L = loglik_matrix(lambda x, th: ens.log_prob(x, th), X, Theta)
    print(f"[ppc-loglik] loglik matrix {L.shape} in {time.time() - t0:.0f}s; "
          f"finite={np.isfinite(L).mean():.4f}", flush=True)

    obs_members = np.zeros((0, Theta.shape[0]), dtype=np.float32)
    if len(obs_paths):
        obs_members = np.stack([loglik_matrix(lambda x, th, m=m: m.forward(x, cond=th), X[:1], Theta)[0]
                                for m in members], 0)

    out = {
        "event_files": np.array([os.path.basename(s) for s in paths]),
        "event_is_obs": is_obs,
        "event_theta_scaled": np.where(is_obs[:, None], np.nan, th_np).astype(np.float32),
        "event_ok": ok,
        "event_x": X.numpy().astype(np.float32),
        "loglik": L,
        "loglik_obs_members": obs_members.astype(np.float32),
        "block_theta_scaled": theta_scaled.astype(np.float64),
        **table,
    }
    overrides = None
    try:
        overrides = _config_preset_overrides(p.config)
    except Exception:
        pass
    prior_status = {}
    for mode in priors:
        try:
            prior, fixed = build_prior_for_mode(mode, param_names, preset_overrides=overrides)
            if fixed is not None:
                raise ValueError("pinning prior modes are not supported for PPC weights")
            lp = prior.log_prob(Theta).detach().cpu().numpy().reshape(-1)
            out[f"logprior_{mode}"] = lp.astype(np.float64)
            prior_status[mode] = {"ok": True, "finite": float(np.isfinite(lp).mean())}
        except Exception as exc:                                  # offline recomputation is possible
            prior_status[mode] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[ppc-loglik] prior {mode} not evaluated ({exc})", flush=True)

    npz = os.path.join(out_dir, stem + ".npz")
    np.savez(npz, **out)
    prov = {"experiment": experiment, "match_string": p.match_string, "theta_store": theta_store,
            "obs_store": obs_store, "null": null, "lock": lock_path, "n_obs": len(obs_paths),
            "n_null": len(null_paths), "n_null_ok": int(ok[~is_obs].sum()), "truth_max_dev": max_dev,
            "theta_table": tt["stats"], "param_names": param_names, "priors": prior_status,
            "likelihood": "grouped" if grouped is not None else "serial",
            "pipeline": p.provenance, "npz": npz}
    with open(os.path.join(out_dir, stem + ".json"), "w") as fh:
        json.dump(_jsonable(prov), fh, indent=2)
    print(f"[ppc-loglik] wrote {npz}", flush=True)
    return prov


def _selftest():
    """Local checks that need no checkpoint: filename parsing, block grouping, null selection, the
    matrix loop against a closed-form Gaussian likelihood."""
    assert parse_mock_name("/a/output_516_out0_rot3_2.h5") == (516, 0, 3, 2)
    assert parse_mock_name("/a/observation_A2.h5") is None
    kept = [f"/s/output_{s}_out{o}_rot{r}_{k}.h5" for s in (1, 2, 3) for o in range(2) for r in range(2) for k in range(4)]
    nul = null_event_paths(kept, [2, 3])
    assert len(nul) == 8 and all(parse_mock_name(p)[3] == 0 and parse_mock_name(p)[0] in (2, 3) for p in nul)
    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.normal(size=(5, 3)), dtype=torch.float32)
    Th = torch.as_tensor(rng.normal(size=(1000, 3)), dtype=torch.float32)
    fn = lambda x, th: -0.5 * ((x - th) ** 2).sum(-1)
    L = loglik_matrix(fn, X, Th, chunk=300, log_every=0)
    ref = -0.5 * ((X[:, None, :] - Th[None]) ** 2).sum(-1).numpy()
    assert np.allclose(L, ref, atol=1e-5), np.abs(L - ref).max()
    print("ppc_loglik selftest OK")


if __name__ == "__main__":
    _selftest()
