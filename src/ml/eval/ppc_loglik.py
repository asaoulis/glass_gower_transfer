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


# --------------------------------------------------------------------------------------------
# cosmology marginal by quadrature
# --------------------------------------------------------------------------------------------
# The design fixes only the N-body cosmology. Its (Om, s8, w0) are the parameters the posterior
# actually constrains; everything else (mnu, h, ns, ombh2 -- prior-dominated -- and the IA
# nuisances) is integrated out of each design point's weight with the NLE itself, so the weights
# are the EXACT analysis-prior marginal p(Om, s8, w0 | t) at the design points and do not inherit
# the Monte-Carlo noise of each sim's 20 random nuisance draws or the design-vs-analysis-prior
# mismatch in h / w (the cause of the pilot's cos-ESS collapse to 1.6; log.md 2026-10-20).
PHI_NAMES = ("omega_m", "sigma_8", "w0")

# Node proposal g per integrated parameter, PHYSICAL units: ('u', lo, hi) or ('n', mean, sd).
# Uniform boxes = the kids_s8_analytic supports (build_s8_box_known_priors / ia_marginal_priors),
# so no node is wasted outside the analysis prior except through its joint (S8, omega_c) box.
NODE_PROPOSALS = {
    "mnu": ("u", 0.06, 0.14), "h": ("u", 0.64, 0.78), "ns": ("u", 0.948, 0.984),
    "ombh2": ("u", 0.022, 0.0228), "b_z": ("n", -3.7, 4.3), "b_src": ("u", -0.5, 1.5),
}
_FORTUNA_MEAN = (5.74, 0.44)
_FORTUNA_SD = (0.29, 0.03)
_FORTUNA_RHO = -0.59


def quadrature_nodes(param_names: Sequence[str], n_nodes: int, seed: int = 0):
    """Fixed (common-random-number) nodes for every non-PHI parameter.

    -> (node_names, nodes [J, K] physical, log_g [J] physical-space proposal log-density).
    Scrambled Sobol, so a 1-D a_ia integral is stratified. NLA-M's (a_ia, b_ia) is drawn from the
    Fortuna Gaussian (= the analysis prior block), NLA / NLA-z / TATT a_ia from U[-6, 6]."""
    from scipy.stats import norm, qmc
    names = [n for n in param_names if n not in PHI_NAMES]
    J = int(n_nodes)
    U = qmc.Sobol(d=len(names), scramble=True, seed=int(seed)).random(J)
    U = np.clip(U, 1e-9, 1 - 1e-9)
    nodes = np.empty((J, len(names)))
    log_g = np.zeros(J)
    nla_m = "b_ia" in names
    for k, n in enumerate(names):
        if nla_m and n in ("a_ia", "b_ia"):
            continue
        spec = NODE_PROPOSALS.get(n, ("u", -6.0, 6.0) if n == "a_ia" else None)
        if spec is None:
            raise ValueError(f"no quadrature proposal for parameter {n!r}")
        if spec[0] == "u":
            nodes[:, k] = spec[1] + U[:, k] * (spec[2] - spec[1])
            log_g += -np.log(spec[2] - spec[1])
        else:
            nodes[:, k] = spec[1] + spec[2] * norm.ppf(U[:, k])
            log_g += norm.logpdf(nodes[:, k], spec[1], spec[2])
    if nla_m:
        ia, ib = names.index("a_ia"), names.index("b_ia")
        z1, z2 = norm.ppf(U[:, ia]), norm.ppf(U[:, ib])
        sa, sb, r = _FORTUNA_SD[0], _FORTUNA_SD[1], _FORTUNA_RHO
        a = _FORTUNA_MEAN[0] + sa * z1
        b = _FORTUNA_MEAN[1] + sb * (r * z1 + np.sqrt(1 - r * r) * z2)
        nodes[:, ia], nodes[:, ib] = a, b
        log_g += (norm.logpdf(z1) + norm.logpdf(z2) - np.log(sa * sb * np.sqrt(1 - r * r)))
    return names, nodes, log_g


@torch.no_grad()
def _quad_marginal(ens, X, phi, param_names, cosmo_scaler, prior_s, node_names, nodes, log_g, *,
                   log_every=200, label="design"):
    """log (1/J) sum_j L(t_e|phi_c, eta_j) pi_s(phi_c, eta_j)/g(eta_j) for events X [E,k] at phi [C,3].
    -> (logmarg [E,C], node_ess [E,C], log-integrand of event 0 [C,J])."""
    iphi = [param_names.index(n) for n in PHI_NAMES]
    inode = [param_names.index(n) for n in node_names]
    C, J, D = phi.shape[0], nodes.shape[0], len(param_names)
    raw = np.empty((C, J, D))
    raw[:, :, iphi] = phi[:, None, :]
    raw[:, :, inode] = nodes[None, :, :]
    scaled = np.asarray(cosmo_scaler.transform(raw.reshape(C * J, D)), dtype=np.float64)
    # g in the SAME scaled space as the priors: scaled density = physical density * prod(box width)
    width = np.asarray(cosmo_scaler.max, np.float64) - np.asarray(cosmo_scaler.min, np.float64)
    log_g_scaled = log_g + float(np.sum(np.log(width[inode])))
    Th = torch.as_tensor(scaled, dtype=torch.float32)
    lps = prior_s.log_prob(Th).detach().cpu().numpy().reshape(C, J).astype(np.float64)
    base = lps - log_g_scaled[None, :]                                   # [C, J]
    live = np.isfinite(base).any(1)                                      # skip phi outside pi_s
    Th_live = Th.reshape(C, J, D)[torch.as_tensor(live)].reshape(-1, D)
    E = int(X.shape[0])
    logmarg = np.full((E, C), -np.inf, dtype=np.float64)
    node_ess = np.zeros((E, C), dtype=np.float32)
    integ0 = None
    t0 = time.time()
    for e in range(E):
        xe = X[e:e + 1]
        Lq = np.empty(Th_live.shape[0], dtype=np.float64)
        for s in range(0, Th_live.shape[0], 16384):
            th = Th_live[s:s + 16384]
            Lq[s:s + th.shape[0]] = ens.log_prob(xe.expand(th.shape[0], -1), th).reshape(-1).cpu().numpy()
        f = np.full((C, J), -np.inf)
        f[live] = Lq.reshape(-1, J) + base[live]                         # log integrand
        with np.errstate(invalid="ignore"):
            m = np.max(f, axis=1, keepdims=True)
            ok = np.isfinite(m[:, 0])
            wj = np.where(ok[:, None], np.exp(f - np.where(ok[:, None], m, 0.0)), 0.0)
            sw = wj.sum(1)
            logmarg[e, ok] = m[ok, 0] + np.log(sw[ok]) - np.log(J)
            node_ess[e] = np.where(ok, sw ** 2 / np.maximum((wj ** 2).sum(1), 1e-300), 0.0)
        if e == 0:
            integ0 = f.astype(np.float32)
        if log_every and (e + 1) % log_every == 0:
            el = time.time() - t0
            print(f"[ppc-loglik] quadrature[{label}] {e + 1}/{E} ({el:.0f}s, ~{el / (e + 1) * (E - e - 1):.0f}s left)",
                  flush=True)
    print(f"[ppc-loglik] quadrature[{label}]: {E} events x {C} phi points ({int(live.sum())} in prior) x {J} "
          f"nodes in {time.time() - t0:.0f}s", flush=True)
    return logmarg, node_ess, integ0


def phi_grid_points(cosmo_scaler, param_names, n: int, seed: int = 1) -> np.ndarray:
    """Scrambled-Sobol points over the scaler's (Om, s8, w0) box (physical units) -- a dense,
    design-independent grid on which the analysis-prior phi-marginal is evaluated (emulator route)."""
    from scipy.stats import qmc
    idx = [list(param_names).index(n_) for n_ in PHI_NAMES]
    lo = np.asarray(cosmo_scaler.min, np.float64)[idx]
    hi = np.asarray(cosmo_scaler.max, np.float64)[idx]
    U = qmc.Sobol(d=3, scramble=True, seed=int(seed)).random(int(n))
    return lo + U * (hi - lo)


@torch.no_grad()
def cosmology_marginals(ens, X, table, param_names, cosmo_scaler, prior_s, prior_g, *,
                        n_nodes: int = 128, seed: int = 0, phi_grid: int = 0,
                        phi_grid_events: int = 0, n_obs_rows: int = 0) -> Dict[str, np.ndarray]:
    """log of  (1/J) sum_j L(t_e | phi_i, eta_j) pi_s(phi_i, eta_j) / g(eta_j)  for every event e and
    design cosmology i  ( = log p(phi_i | t_e) + log Z_e, the analysis-prior marginal at phi_i ),
    plus log q_D(phi_i), the design density in phi (the Gower prior with every non-PHI parameter
    pinned to a fixed reference value: build_gower_prior factorises, so this is q_D(phi) times a
    constant common to all i). Optionally the same marginal on a dense phi grid for the first
    `phi_grid_events` events (the observation first)."""
    param_names = list(param_names)
    iphi = [param_names.index(n) for n in PHI_NAMES]
    sims, first = np.unique(table["block_sim"], return_index=True)
    phi = table["block_theta_raw"][first][:, iphi]                       # [C, 3] physical
    node_names, nodes, log_g = quadrature_nodes(param_names, n_nodes, seed)
    inode = [param_names.index(n) for n in node_names]
    logmarg, node_ess, integ0 = _quad_marginal(ens, X, phi, param_names, cosmo_scaler, prior_s,
                                               node_names, nodes, log_g)

    # design density: Gower prior at (phi_i, reference eta)
    ref = table["block_theta_raw"][first].copy()
    ref_vals = {"mnu": 0.1, "h": 0.7022, "ns": 0.9649, "ombh2": 0.02237, "a_ia": 5.74, "b_ia": 0.44,
                "b_z": -3.7, "b_src": 0.5}
    if "b_ia" not in node_names:
        ref_vals["a_ia"] = 0.0                                           # inside U[-6, 6]
    for k, n in zip(inode, node_names):
        ref[:, k] = ref_vals.get(n, 0.0)
    lq = prior_g.log_prob(torch.as_tensor(cosmo_scaler.transform(ref), dtype=torch.float32))
    lq = lq.detach().cpu().numpy().reshape(-1).astype(np.float64)
    out = {"cos_sim": sims.astype(np.int32), "cos_phi_raw": phi, "cos_logq_design": lq,
           "logmarg": logmarg.astype(np.float32), "node_ess": node_ess,
           "node_names": np.array(node_names), "nodes_raw": nodes, "nodes_log_g": log_g,
           "integrand_event0": integ0}
    if phi_grid and phi_grid_events:
        G = phi_grid_points(cosmo_scaler, param_names, phi_grid)
        # the observation (row 0 when given) + a STRIDED subset of the null events, so the grid null
        # spans every cosmology (the null list is sorted by filename = by sim).
        E = int(X.shape[0])
        n_obs = int(n_obs_rows)
        k = max(0, min(int(phi_grid_events) - n_obs, E - n_obs))
        null_idx = np.arange(n_obs, E)
        if k and null_idx.size > k:
            null_idx = null_idx[np.linspace(0, null_idx.size - 1, k).round().astype(int)]
        gidx = np.r_[np.arange(n_obs), null_idx[:k]].astype(int)
        lg, ng, _ = _quad_marginal(ens, X[gidx], G, param_names, cosmo_scaler, prior_s,
                                   node_names, nodes, log_g, log_every=50, label="grid")
        out.update({"grid_phi_raw": G, "grid_logmarg": lg.astype(np.float32),
                    "grid_node_ess": ng, "grid_event_index": gidx})
    return out


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
    n_nodes: int = 128,
    node_seed: int = 0,
    phi_grid: int = 2048,
    phi_grid_events: int = 301,
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

    # ---- cosmology marginal by quadrature (the PPC cosmology weights; plan.md §M3 rev. 2) -------
    overrides = None
    try:
        overrides = _config_preset_overrides(p.config)
    except Exception:
        pass
    quad = {}
    if n_nodes and n_nodes > 0:
        prior_s, fx = build_prior_for_mode("kids_s8_analytic", param_names, preset_overrides=overrides)
        prior_g, _ = build_prior_for_mode("gower", param_names, preset_overrides=overrides)
        quad = cosmology_marginals(ens, X, table, param_names, cosmo_scaler, prior_s, prior_g,
                                   n_nodes=int(n_nodes), seed=int(node_seed), phi_grid=int(phi_grid or 0),
                                   phi_grid_events=int(phi_grid_events or 0), n_obs_rows=len(obs_paths))

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
        **quad,
    }
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
            "quadrature": {"n_nodes": int(n_nodes or 0), "node_seed": int(node_seed),
                           "node_names": list(quad.get("node_names", []))},
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
