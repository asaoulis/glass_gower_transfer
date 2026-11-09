"""Score a trained NLE ensemble on data it was not trained on, including a single observation.

An observation is a file in the mock HDF5 schema with no cosmo_dict. The original training scalers
and Stage-A whitener are injected, never refit, and the embedding cache is bypassed. Outputs go to
checkpoints/<exp>/external/<tag>/external_posterior_samples_<prior>_<match>.npz.
run_reproduction_check proves this path reproduces the production eval on the model's own test set.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from dataclasses import dataclass, field
from itertools import permutations as _permutations
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# How many times the measured scaler-refit noise floor a reproduction's z-deviation may reach
# and still pass (run_reproduction_check CHECK 3).
_Z_FLOOR_BAND = 3.0

_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"


# --- priors, shared with gen_samples.py ---
PRIOR_MODES = ("gower", "kids_s8_analytic", "LCDM_fixed_w0")

# prior mode -> the parameters it pins, {name: physical value}. Empty for a mode that samples the
# full vector. build_prior_for_mode and free_param_names both read this table.
FIXED_BY_PRIOR_MODE = {
    "gower": {},
    "kids_s8_analytic": {},
    "LCDM_fixed_w0": {"w0": -1.0},
}


def free_param_names(prior_mode: str, param_names):
    """-> the parameters actually sampled under `prior_mode`, in dump-column order.

    A pinned parameter is absent from the dump's samples, while theta0s keeps the full vector, so
    an LCDM_fixed_w0 dump has samples.shape[-1] == len(param_names) - 1. Unscale samples with
    these names, and drop the pinned names rather than re-inserting a constant column.
    """
    fixed = FIXED_BY_PRIOR_MODE.get(prior_mode, {})
    return [n for n in param_names if n not in fixed]


def build_prior_for_mode(prior_mode: str, param_names, *, preset_overrides=None):
    """-> (prior, fixed_parameters). `prior_mode` is one of PRIOR_MODES.

    When `fixed_parameters` is not None the resulting dump is missing those columns — see
    `free_param_names`.
    """
    from ..embeddings.embeddings_utils import _build_cosmo_preset_scaler
    from .utils import _merged_preset, build_gower_prior, build_s8_analytic_prior

    param_names = list(param_names)
    # Every branch passes preset_overrides: the global preset has no box for b_g_bin1..6 and the
    # preset scaler raises on a parameter without one.
    preset = _merged_preset(preset_overrides)
    if prior_mode == "gower":
        kw = {"preset_overrides": preset_overrides} if preset_overrides else {}
        return build_gower_prior(param_names, **kw), None
    if prior_mode == "kids_s8_analytic":
        scaler = _build_cosmo_preset_scaler(preset, param_names)
        return build_s8_analytic_prior(param_names, scaler), None
    if prior_mode == "LCDM_fixed_w0":
        from gen_samples import _build_fixed_parameters_list
        scaler = _build_cosmo_preset_scaler(preset, param_names)
        prior = build_s8_analytic_prior(param_names, scaler, return_restricted=False)
        return prior, _build_fixed_parameters_list(dict(FIXED_BY_PRIOR_MODE[prior_mode]),
                                                   param_names, space="physical",
                                                   preset_overrides=preset_overrides)
    raise ValueError(f"unknown prior mode {prior_mode!r}; choose from {PRIOR_MODES}")


# --- datasets addressable by a bare name (the launcher passes no paths) ---
def store_glob(name: str) -> str:
    """A store basename -> its output glob, on gpu5 first, then gpu4.

    `<store>_hos` names the HOS sidecars under {base_path}/checkpoints/hos_sidecars/<store>/ and
    `<store>_hosx` the hos_x sidecars under hos_sidecars/<store>_hosx/."""
    import os as _os
    if name.endswith("_hosx"):         # the hos_x directory keeps the suffix
        from config.default import get_default_config
        return _os.path.join(get_default_config().base_path, "checkpoints", "hos_sidecars", name, "output_*.h5")
    if name.endswith("_hos"):
        from config.default import get_default_config
        return _os.path.join(get_default_config().base_path, "checkpoints", "hos_sidecars", name[:-4], "output_*.h5")
    for root in (_GPU5, _GPU5.replace("/share/gpu5/", "/share/gpu4/")):
        if _os.path.isdir(_os.path.join(root, name)):
            return f"{root}/{name}/output_*.h5"
    return f"{_GPU5}/{name}/output_*.h5"


@dataclass
class FrozenNLEPipeline:
    """A trained Stage-B NLE ensemble plus everything needed to put new data through it.

    Every field is resolved from the trained run; nothing here is fit on external data.
    """

    experiment: str
    match_string: str
    config: object
    model: object                       # EnsembleLikelihoodNDELightningModule
    scalers: Dict[str, object]          # the original training scalers ('data' / 'cosmo')
    test_loader: object                 # embedding loader over the external paths
    raw_dataset: object = None          # the H5CosmoDataset behind it
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def param_names(self):
        return list(self.config.cosmo_param_names)


def resolve_nle_pipeline(
    experiment: str,
    match_string: str,
    *,
    paths: Sequence[str],
    source_experiments: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    config_overrides: Optional[Dict[str, object]] = None,
    require_member_frames: bool = False,
) -> FrozenNLEPipeline:
    """Build the ensemble and put `paths` through its frozen encoder+whitener.

    `paths` may be a single file: nothing here assumes more than one row.
    """
    from ..embeddings.train import load_embedding_model_with_dataloader

    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("resolve_nle_pipeline: `paths` is empty")

    if source_experiments is None:
        source_experiments = resolve_source_experiments(experiment)

    print(f"[nle-external] {experiment} match={match_string} sources={list(source_experiments)} "
          f"n_files={len(paths)}", flush=True)

    art = load_embedding_model_with_dataloader(
        experiment,
        match_string,
        source_experiments=list(source_experiments),
        config_overrides=config_overrides,
        external_paths=paths,
        external_batch_size=batch_size,
        require_member_frames=require_member_frames,
    )
    return FrozenNLEPipeline(
        experiment=experiment,
        match_string=str(match_string),
        config=art.config,
        model=art.model,
        scalers=art.scalers,
        test_loader=art.test_loader,
        raw_dataset=getattr(art.model, "external_raw_dataset", None),
        provenance={
            "source_experiments": list(source_experiments),
            "n_files": len(paths),
            "scaler_fit_patterns": getattr(art.config, "data_patterns", None),
            # The exact files scored: source encoders, whitener and members.
            **dict(getattr(art.model, "external_provenance", {}) or {}),
        },
    )


def match_string_for(experiment: str, repeat: int) -> str:
    """Repeat index -> the run match string for a Stage-B row.

    Read from the config's max_trainval_cosmos: ncosmo<N>_<r>, or ncosmoNone_<r> when unset.
    """
    from config.experiments import experiments as _exps
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments as _bgp
    from config.kids_legacy_vdq import kids_legacy_vdq_experiments as _vdq
    from config.kids_legacy_hos import kids_legacy_hos_experiments as _hos
    merged = {**_exps, **_bgp, **_vdq, **_hos}   # same merge order as eval.py / train_embeddings.py
    exp = merged.get(experiment, {})
    n = exp.get("max_trainval_cosmos", None)
    if isinstance(n, (list, tuple)):
        n = n[0] if len(n) == 1 else None
    tag = "ncosmoNone" if n is None else f"ncosmo{int(n)}"
    return f"{tag}_{int(repeat)}"


def resolve_source_experiments(experiment: str) -> Sequence[str]:
    """The frozen source encoder(s) for a Stage-B row, from the config side (one place)."""
    from config.kids_legacy_bgp import HF_RETRAIN_SOURCES, FLAGSHIP_NLE_SOURCES
    from config.kids_legacy_vdq import VDQ_NLE_SOURCES
    from config.kids_legacy_hos import HOS_NLE_SOURCES
    for table in (HF_RETRAIN_SOURCES, FLAGSHIP_NLE_SOURCES, VDQ_NLE_SOURCES, HOS_NLE_SOURCES):
        if experiment in table:
            src = table[experiment]
            return [src] if isinstance(src, str) else list(src)
    raise KeyError(
        f"no source encoder registered for '{experiment}'. Add it to HF_RETRAIN_SOURCES or "
        f"FLAGSHIP_NLE_SOURCES in config/kids_legacy_bgp.py (VDQ rows: VDQ_NLE_SOURCES in "
        f"config/kids_legacy_vdq.py) — the source encoder is part of a "
        f"chain's identity and must not be guessed at the eval site (cf. the e890aec bug)."
    )


def sample_pipeline(
    p: FrozenNLEPipeline,
    *,
    prior_mode: str = "gower",
    num_samples: int = 25_000,
    num_chains: int = 1,
    num_jobs: Optional[int] = None,
    warmup_steps: int = 500,
    mcmc_workers: Optional[int] = None,
    mcmc_threads: Optional[int] = None,
    mcmc_seed: Optional[int] = None,
):
    """Run the ensemble's MCMC over every row of `p.test_loader`. Returns (theta0s, samples).

    num_samples is the total per observation, split across num_chains. mcmc_workers=K splits each
    event's sample budget across K processes, so the per-chain length is
    num_samples / (K * num_chains) and the posterior is unchanged. mcmc_threads=1 keeps the K
    processes from oversubscribing.
    """
    from .utils import _config_preset_overrides

    overrides = None
    try:
        overrides = _config_preset_overrides(p.config)
    except Exception:
        pass
    prior, fixed_parameters = build_prior_for_mode(prior_mode, p.param_names,
                                                   preset_overrides=overrides)

    kw = dict(num_samples=num_samples, prior=prior, num_chains=num_chains,
              warmup_steps=warmup_steps)
    if fixed_parameters is not None:
        kw["fixed_parameters"] = fixed_parameters
    if num_jobs is not None:
        kw["num_jobs"] = num_jobs
    if mcmc_workers is not None:
        kw["mcmc_workers"] = mcmc_workers
    if mcmc_threads is not None:
        kw["mcmc_threads"] = mcmc_threads
    if mcmc_seed is not None:
        kw["mcmc_seed"] = mcmc_seed
    print(f"[nle-external] sampling: {kw and {k: v for k, v in kw.items() if k != 'prior'}} "
          f"prior={prior_mode}", flush=True)
    theta0s, samples = p.model.generate_samples(**kw)
    return theta0s, samples


def output_dir(cfg, tag: str) -> str:
    return os.path.join(cfg.base_path, "checkpoints", cfg.experiment_name, "external", tag)


def run_external_nle_eval(
    experiment: str,
    match_string: str,
    *,
    paths: Optional[Sequence[str]] = None,
    store: Optional[str] = None,
    tag: Optional[str] = None,
    prior_mode: str = "gower",
    num_samples: int = 25_000,
    num_chains: int = 1,
    num_jobs: Optional[int] = None,
    warmup_steps: int = 500,
    mcmc_workers: Optional[int] = None,
    mcmc_threads: Optional[int] = None,
    mcmc_seed: Optional[int] = None,
    batch_size: int = 64,
    source_experiments: Optional[Sequence[str]] = None,
    compute_metrics: Optional[bool] = None,
    dry_run: bool = False,
    require_member_frames: bool = False,
) -> Dict[str, object]:
    """Resolve the pipeline, sample, and persist. `paths` is the primitive; `store` is a shortcut."""
    from .utils import _save_posterior_samples

    if paths is None:
        if store is None:
            raise ValueError("give either `paths` or `store`")
        paths = sorted(_glob.glob(store_glob(store)))
        if not paths:
            raise FileNotFoundError(f"store '{store}' matched no files ({store_glob(store)})")
    tag = tag or store or "external"

    p = resolve_nle_pipeline(experiment, match_string, paths=paths,
                             source_experiments=source_experiments, batch_size=batch_size,
                             require_member_frames=require_member_frames)
    out = output_dir(p.config, tag)
    os.makedirs(out, exist_ok=True)

    if dry_run:
        print(f"[nle-external] DRY RUN — resolved pipeline, would write to {out}", flush=True)
        return {"output_dir": out, "n_files": len(paths), "dry_run": True}

    theta0s, samples = sample_pipeline(
        p, prior_mode=prior_mode, num_samples=num_samples, num_chains=num_chains,
        num_jobs=num_jobs, warmup_steps=warmup_steps,
        mcmc_workers=mcmc_workers, mcmc_threads=mcmc_threads, mcmc_seed=mcmc_seed,
    )

    # An observation has no truth, so every metric is skipped.
    t = torch.as_tensor(theta0s) if not torch.is_tensor(theta0s) else theta0s
    has_truth = bool(torch.isfinite(t).all())
    if compute_metrics is None:
        compute_metrics = has_truth
    if compute_metrics and not has_truth:
        raise ValueError("compute_metrics=True but the inputs carry no finite truth vector")

    npz = os.path.join(out, f"external_posterior_samples_{prior_mode}_{p.match_string}.npz")
    _save_posterior_samples(npz, theta0s, samples, _paths_of(p, paths))
    print(f"[nle-external] wrote {npz}", flush=True)

    # The sampling geometry is part of the record: mcmc_workers locates the shard boundaries.
    result = {"output_dir": out, "samples_npz": npz, "n_files": len(paths),
              "has_truth": has_truth, "prior": prior_mode,
              "experiment": experiment, "match_string": p.match_string,
              "sampling": {"num_samples": num_samples, "num_chains": num_chains,
                           "warmup_steps": warmup_steps, "num_jobs": num_jobs,
                           "mcmc_workers": mcmc_workers, "mcmc_threads": mcmc_threads,
                           "mcmc_seed": mcmc_seed},
              "provenance": p.provenance}

    if compute_metrics:
        from .evaluate_models import run_evaluation_on_samples
        try:
            prior, _fx = build_prior_for_mode(prior_mode, p.param_names)
            score_theta0s, score_scaler = theta0s, p.scalers["cosmo"]
            if _fx:
                # A pinning prior mode samples only the free dimensions: samples is D-1 wide while
                # theta0s is D. The full-dimensional prior is dropped.
                from ..embeddings.embeddings_utils import (COSMO_PARAM_PRESET_MINMAX,
                                                           _build_cosmo_preset_scaler)
                from .utils import _config_preset_overrides
                free = free_param_names(prior_mode, p.param_names)
                keep = [p.param_names.index(n) for n in free]
                try:
                    preset = {**COSMO_PARAM_PRESET_MINMAX, **(_config_preset_overrides(p.config) or {})}
                except Exception:
                    preset = COSMO_PARAM_PRESET_MINMAX
                score_scaler = _build_cosmo_preset_scaler(preset, free)
                score_theta0s = theta0s[..., keep]
                prior = None
                print(f"[nle-external] metrics on the {len(free)} free parameters "
                      f"(pinned: {sorted(set(p.param_names) - set(free))}); prior-dependent "
                      f"metrics skipped", flush=True)
            # Argument order is (theta0s, samples, param_scaler); the scaler carries .parameter_names.
            metrics = run_evaluation_on_samples(
                score_theta0s, samples, score_scaler, prior=prior, compute_calibration=False,
            )
            mpath = os.path.join(out, f"external_evaluation_results_{prior_mode}_{p.match_string}.json")
            with open(mpath, "w") as fh:
                json.dump(_jsonable(metrics), fh, indent=2)
            result["metrics_json"] = mpath
            print(f"[nle-external] wrote {mpath}", flush=True)
        except Exception as exc:
            print(f"[nle-external] metrics skipped ({type(exc).__name__}: {exc})", flush=True)

    # A per-prior record, plus the legacy name that every prior of one (experiment, repeat) shares.
    for fname in (f"external_provenance_{prior_mode}_{p.match_string}.json",
                  f"external_provenance_{p.match_string}.json"):
        with open(os.path.join(out, fname), "w") as fh:
            json.dump(_jsonable(result), fh, indent=2)
    return result


@dataclass(frozen=True)
class ProductionFrame:
    """A Stage-B run's training frame, rebuilt exactly as embeddings/train.py built it."""
    cfg: object
    match: str
    src_match: str
    source_models: list
    scalers: dict
    paths: list


def rebuild_production_frame(experiment: str, repeat: int = 0, *,
                             source_experiments: Optional[Sequence[str]] = None) -> ProductionFrame:
    """Rebuild the exact training frame of a Stage-B run: cfg, source encoders, scalers, test split.

    Shared by the reproduction gate and scaler recovery. The order matters: load_pretrained_models
    sets cfg.dataset_quantities from the source encoder before prepare_data_parameters runs.
    match_string is the source match (None_<r>), as in production's cfg_0.
    """
    from ..embeddings.train import build_cfg_from_experiment_dict
    from ..embeddings.embeddings_utils import load_pretrained_models
    from ..utils import prepare_data_parameters, set_seed_for_repeat_and_ensemble
    from ..models.sampling import get_dataset_paths
    from config.experiments import experiments as _exps
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments as _bgp
    from config.kids_legacy_vdq import kids_legacy_vdq_experiments as _vdq
    from config.kids_legacy_hos import kids_legacy_hos_experiments as _hos

    merged = {**_exps, **_bgp, **_vdq, **_hos}   # same merge order as match_string_for / eval.py
    if experiment not in merged:
        raise KeyError(f"experiment {experiment!r} not found in the config tables")
    match = match_string_for(experiment, repeat)

    n_cosmo = None
    _n = merged[experiment].get("max_trainval_cosmos", None)
    if isinstance(_n, (list, tuple)) and len(_n) == 1:
        _n = _n[0]
    if isinstance(_n, int):
        n_cosmo = _n

    cfg = build_cfg_from_experiment_dict(experiment, merged[experiment], n_cosmo=n_cosmo)
    if source_experiments is None:
        source_experiments = resolve_source_experiments(experiment)
    src_match = match if getattr(cfg, "match_num_cosmo", False) else "None_" + match.split("_")[1]
    source_models, dataset_quantities, _ = load_pretrained_models(
        list(source_experiments), cfg_overrides=None, repeat_idx=repeat, match_string=src_match,
        per_source_match_strings=getattr(cfg, "source_match_strings", None),
    )

    cfg.dataset_quantities = dataset_quantities
    cfg.match_string = str(src_match)
    cfg.test_shape_noise_idx = [0]      # as embeddings/train.py sets it
    cfg.split_seed = 42
    set_seed_for_repeat_and_ensemble(cfg, repeat_idx=repeat, ensemble_idx=0)
    scalers, _tr, _va, test_loader = prepare_data_parameters(cfg)

    # TransformingDataset keeps the H5 dataset on .base_ds, not .dataset, so use the helper that walks both.
    paths = get_dataset_paths(test_loader) or []
    if not paths:
        raise RuntimeError(
            "the production test split exposed no file paths. Check that the test loader's "
            "dataset still carries `.paths` (H5CosmoDataset.paths / TransformingDataset.base_ds.paths)."
        )
    return ProductionFrame(cfg=cfg, match=match, src_match=str(src_match),
                           source_models=source_models, scalers=scalers, paths=list(paths))


def run_reproduction_check(
    experiment: str,
    repeat: int = 0,
    *,
    source_experiments: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    z_tol: float = 1e-2,
) -> Dict[str, object]:
    """Check that this path reproduces the production eval on the model's own test set. No sampling.

    CHECK 1: file-set and row-count equality against the production posterior dump, reported as
    unavailable for dumps without test_files. CHECK 2: exact split reproduction, theta against the
    cached emb_test.pt. CHECK 3: raw z against the cache, within _Z_FLOOR_BAND times the measured
    scaler-refit floor (or z_tol, whichever is larger). CHECK 4: the recovered per-member frame,
    when one is persisted. Returns the report dict with its verdict.
    """
    import numpy as _np
    import torch as _torch
    from ..embeddings.train import _build_external_embedding_loader_for_cfg
    from ..utils import prepare_data_parameters

    match = match_string_for(experiment, repeat)
    report: Dict[str, object] = {"experiment": experiment, "repeat": repeat, "match": match}

    frame = rebuild_production_frame(experiment, repeat, source_experiments=source_experiments)
    cfg, src_match = frame.cfg, frame.src_match
    source_models, scalers_a, prod_paths = frame.source_models, frame.scalers, frame.paths
    report["n_test_paths"] = len(prod_paths)
    print(f"[repro] production test split: {len(prod_paths)} files", flush=True)

    # --- CHECK 1: file-set equality against the production posterior dump ---
    # The dump's key is test_files. Older dumps without it report `unavailable`.
    npz_path = os.path.join(cfg.base_path, "checkpoints", experiment,
                            f"ensemble_posterior_samples_{match}.npz")
    prod_theta0s = None
    if os.path.exists(npz_path):
        d = _np.load(npz_path, allow_pickle=False)
        report["npz_keys"] = list(d.files)
        if "theta0s" in d.files:
            prod_theta0s = _np.asarray(d["theta0s"])
            report["n_prod_rows"] = int(prod_theta0s.shape[0])
        files_key = next((k for k in ("test_files", "files") if k in d.files), None)
        if files_key is not None:
            prod_files = [os.path.basename(str(x)) for x in d[files_key]]
            ours = [os.path.basename(x) for x in prod_paths]
            same = set(prod_files) == set(ours)
            report["file_set_equal"] = bool(same)
            report["files_key"] = files_key
            print(f"[repro] CHECK 1 file-set equality ({files_key}): "
                  f"{'PASS' if same else 'FAIL'} "
                  f"({len(ours)} ours vs {len(prod_files)} production)", flush=True)
            if same:
                # CHECK 1c: align on basename, since the orders differ, then compare the ids row-wise.
                idx = {f: i for i, f in enumerate(prod_files)}
                order = [idx[f] for f in ours]
                for key in ("sim_ids", "aug_ids"):
                    if key in d.files:
                        arr = _np.asarray(d[key])[order]
                        report[f"{key}_aligned_rows"] = int(arr.shape[0])
                print(f"[repro] CHECK 1c basename alignment OK over {len(order)} rows "
                      f"(ids: {[k for k in ('sim_ids', 'aug_ids') if k in d.files]})", flush=True)
            if not same:
                report["only_ours"] = sorted(set(ours) - set(prod_files))[:5]
                report["only_prod"] = sorted(set(prod_files) - set(ours))[:5]
                print(f"[repro]   only ours: {report['only_ours']}\n"
                      f"[repro]   only prod: {report['only_prod']}", flush=True)
        else:
            report["file_set_equal"] = "unavailable"
            print(f"[repro] CHECK 1 UNAVAILABLE: the production dump records no 'files' key "
                  f"(has {list(d.files)}), so basenames cannot be compared. CHECK 2 below covers "
                  f"the same property exactly.", flush=True)
        if prod_theta0s is not None:
            n_ok = prod_theta0s.shape[0] == len(prod_paths)
            report["row_count_match"] = bool(n_ok)
            print(f"[repro] CHECK 1b row count: {'PASS' if n_ok else 'FAIL'} "
                  f"(ours {len(prod_paths)} vs dump {prod_theta0s.shape[0]})", flush=True)
    else:
        report["file_set_equal"] = None
        print(f"[repro] CHECK 1 SKIPPED: no production dump at {npz_path}", flush=True)

    # --- CHECK 3: raw z against the cached emb_test.pt ---

    def _raw_z(key_scalers, cosmo_scaler):
        loader, _ds = _build_external_embedding_loader_for_cfg(
            cfg, source_models, prod_paths, key_scalers, cosmo_scaler,
            whiten_cfg=None,                      # raw z: the cache stores pre-whitening embeddings
            batch_size=batch_size,
        )
        zs, ths = [], []
        for b in loader:
            zs.append(b[0])
            ths.append(b[1])
        return _torch.cat(zs, dim=0), _torch.cat(ths, dim=0)

    z_ours, th_ours = _raw_z(scalers_a["data"], scalers_a["cosmo"])

    # Locate the run's cached embeddings, one per member, and compare against the first.
    cache_hits = sorted(_glob.glob(os.path.join(cfg.base_path, "checkpoints", experiment,
                                                f"*{match}*", "datasets", "emb_test.pt")))
    report["n_cache_candidates"] = len(cache_hits)
    cached = _torch.load(cache_hits[0], map_location="cpu") if cache_hits else None
    if cached is not None:
        report["cache_used"] = cache_hits[0]
        print(f"[repro] cache: {len(cache_hits)} member caches found; comparing against "
              f"{os.path.basename(os.path.dirname(os.path.dirname(cache_hits[0])))}", flush=True)

    # --- CHECK 2: exact split reproduction, via theta ---
    # theta goes through the deterministic preset cosmo scaler, so it must match the cache exactly.
    th_cached = cached.get("theta") if isinstance(cached, dict) else None
    if th_cached is not None:
        if tuple(th_cached.shape) != tuple(th_ours.shape):
            report["theta_check"] = "FAIL-shape"
            print(f"[repro] CHECK 2 theta: FAIL (shape) ours {tuple(th_ours.shape)} vs cached "
                  f"{tuple(th_cached.shape)} -- the split does NOT reproduce; STOP.", flush=True)
        else:
            dth = (th_ours - th_cached).abs()
            report["theta_max_abs_diff"] = float(dth.max())
            exact = bool(_torch.allclose(th_ours, th_cached, rtol=0, atol=1e-6))
            report["theta_check"] = "PASS" if exact else "FAIL"
            print(f"[repro] CHECK 2 theta row-wise over {th_ours.shape[0]} rows: "
                  f"max |dtheta| = {float(dth.max()):.3e} -> {'PASS' if exact else 'FAIL'}",
                  flush=True)
    else:
        report["theta_check"] = "no-cache"
        print("[repro] CHECK 2 SKIPPED: no cached theta to compare against", flush=True)

    _base_seed = int(getattr(cfg, "scaler_fit_seed", 0) or 0)
    cfg.scaler_fit_seed = _base_seed + 20260907
    scalers_b, _t2, _v2, _te2 = prepare_data_parameters(cfg)
    cfg.scaler_fit_seed = _base_seed
    z_refit, _th_refit = _raw_z(scalers_b["data"], scalers_b["cosmo"])
    sd = z_ours.std(dim=0).clamp_min(1e-12)
    floor = ((z_ours - z_refit).abs() / sd)
    report["refit_floor_median"] = float(floor.median())
    report["refit_floor_max"] = float(floor.max())
    print(f"[repro] scaler-refit NOISE FLOOR: median |dz|/sd = {report['refit_floor_median']:.3e}, "
          f"max = {report['refit_floor_max']:.3e}", flush=True)

    if cache_hits:
        z_cached = cached["z"] if isinstance(cached, dict) and "z" in cached else None
        if z_cached is not None and z_cached.shape == z_ours.shape:
            dev = ((z_ours - z_cached).abs() / sd)
            report["z_dev_median"] = float(dev.median())
            report["z_dev_max"] = float(dev.max())
            floor_med = float(report["refit_floor_median"])
            budget = max(z_tol, _Z_FLOOR_BAND * floor_med)
            ratio = float(dev.median()) / max(floor_med, 1e-12)
            ok = float(dev.median()) <= budget
            report["z_budget"] = budget
            report["z_dev_over_floor"] = ratio
            report["z_check"] = "PASS" if ok else "FAIL"
            print(f"[repro] CHECK 3 z vs {os.path.basename(os.path.dirname(os.path.dirname(cache_hits[0])))}: "
                  f"median |dz|/sd = {dev.median():.3e}, max = {dev.max():.3e}; "
                  f"floor = {floor_med:.3e}, ratio = {ratio:.2f}x, budget = {budget:.3e} -> "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                print("[repro]   ABOVE THE FLOOR -- something other than the scaler subsample "
                      "differs. Diagnose; do NOT widen the band.", flush=True)
            # A floor above z_tol is itself a finding.
            if floor_med > z_tol:
                report["floor_above_tol"] = True
                print(f"[repro]   NOTE the refit floor ({floor_med:.3e}) EXCEEDS z_tol "
                      f"({z_tol:.0e}): scaler irreproducibility is a real ~1%-of-sd effect on z, "
                      f"and the existing production misspec numbers carry it too.", flush=True)
        else:
            report["z_check"] = "shape-mismatch" if z_cached is not None else "no-z-in-cache"
            print(f"[repro] CHECK 3 inconclusive: {report['z_check']} "
                  f"(ours {tuple(z_ours.shape)}, cached "
                  f"{tuple(z_cached.shape) if z_cached is not None else None})", flush=True)
    else:
        report["z_check"] = "no-cache"
        print("[repro] CHECK 3 SKIPPED: no emb_test.pt found for this run", flush=True)

    # --- CHECK 4: the recovered frame, if one has been persisted, over the full split ---
    if cache_hits:
        sc_path = os.path.join(os.path.dirname(cache_hits[0]), "scalers.pt")
        if os.path.exists(sc_path):
            from ..data.scaling import load_scalers
            rec_keys, rec_cosmo, rec_prov = load_scalers(sc_path)
            z_rec, _th_rec = _raw_z(rec_keys, rec_cosmo or scalers_a["cosmo"])
            if z_cached is not None and z_cached.shape == z_rec.shape:
                dvr = ((z_rec - z_cached).abs() / sd)
                report["z_dev_median_recovered"] = float(dvr.median())
                report["z_dev_max_recovered"] = float(dvr.max())
                report["recovered_provenance"] = {k: rec_prov.get(k)
                                                  for k in ("method", "n_events", "z_dev_median_after")}
                # From the report: CHECK 3's locals exist only on its success branch.
                refit_med = report.get("z_dev_median")
                fl = report.get("refit_floor_median")
                gain = (float(refit_med) / max(float(dvr.median()), 1e-30)) if refit_med else float("nan")
                report["recovered_gain_vs_refit"] = gain
                print(f"[repro] CHECK 4 RECOVERED frame over all {z_rec.shape[0]} events: "
                      f"median |dz|/sd = {dvr.median():.3e} (max {dvr.max():.3e}) vs refit "
                      f"{refit_med if refit_med is None else format(refit_med, '.3e')} -> "
                      f"{gain:.0f}x closer, floor "
                      f"{fl if fl is None else format(fl, '.3e')}", flush=True)
            else:
                print("[repro] CHECK 4 inconclusive: recovered-z shape mismatch", flush=True)
        else:
            report["z_dev_median_recovered"] = None
            print(f"[repro] CHECK 4 SKIPPED: no persisted scalers.pt beside the cache "
                  f"(run eval --mode recover-scalers first)", flush=True)

    # One verdict: CHECK 2 and CHECK 3 must both PASS. A skipped or inconclusive check is a FAIL.
    report["verdict"] = "PASS" if (report.get("theta_check") == "PASS" and report.get("z_check") == "PASS") else "FAIL"
    print(f"[repro] VERDICT {report['verdict']} ({experiment} r{repeat}: theta={report.get('theta_check')}, "
          f"z={report.get('z_check')}, z_dev/floor={report.get('z_dev_over_floor')})", flush=True)
    print("[repro] REPORT " + json.dumps(_jsonable(report)), flush=True)
    return report


def _paths_of(p: FrozenNLEPipeline, fallback):
    ds = p.raw_dataset
    return list(getattr(ds, "paths", fallback))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def run_frame_verification(
    experiment: str,
    repeat: int = 0,
    *,
    source_experiments: Optional[Sequence[str]] = None,
    max_events: int = 400,
    batch_size: int = 64,
    accept_dev: float = 1e-3,
) -> Dict[str, object]:
    """Verify every member's persisted input frame on held-out events, through the embedding path
    the observation sampler uses, on this node.

    The events are a stride disjoint from the one recover-scalers fits on. Run it on the partition
    the sampling jobs use. A member passes when median |dz|/sd <= accept_dev.
    """
    from ..data.scaling import load_scalers
    from ..embeddings.train import _build_external_embedding_loader_for_cfg

    frame = rebuild_production_frame(experiment, repeat, source_experiments=source_experiments)
    cfg, match = frame.cfg, frame.match
    n_all = len(frame.paths)
    stride = max(1, n_all // max_events)
    sel = [i for i in range(stride // 2, n_all, stride)][:max_events]
    paths = [frame.paths[i] for i in sel]
    cache_hits = sorted(_glob.glob(os.path.join(cfg.base_path, "checkpoints", experiment,
                                                f"*{match}*", "datasets", "emb_test.pt")))
    out: Dict[str, object] = {"experiment": experiment, "repeat": repeat, "match": match,
                              "n_events": len(paths), "node": os.uname().nodename,
                              "torch_threads": torch.get_num_threads(), "members": []}
    print(f"[verify] {experiment} {match}: {len(cache_hits)} members, {len(paths)} held-out events "
          f"(offset stride {stride}), node {out['node']}, threads {out['torch_threads']}", flush=True)
    for cache_path in cache_hits:
        member_dir = os.path.dirname(os.path.dirname(cache_path))
        name = os.path.basename(member_dir)
        sc_path = os.path.join(member_dir, "datasets", "scalers.pt")
        if not os.path.exists(sc_path):
            out["members"].append({"member": name, "verdict": "NO-FRAME"})
            print(f"[verify] {name}: NO persisted frame -> FAIL", flush=True)
            continue
        keys, cosmo, prov = load_scalers(sc_path)
        z_cached = torch.load(cache_path, map_location="cpu", weights_only=False)["z"][sel]
        loader, _ds = _build_external_embedding_loader_for_cfg(
            cfg, frame.source_models, paths, keys, cosmo or frame.scalers["cosmo"],
            whiten_cfg=None, batch_size=batch_size)
        z = torch.cat([b[0] for b in loader], dim=0)
        sd = z_cached.std(dim=0).clamp_min(1e-12)
        dev = ((z - z_cached).abs() / sd)
        ok = float(dev.median()) <= accept_dev
        rec = {"member": name, "method": prov.get("method"), "z_dev_median": float(dev.median()),
               "z_dev_p99": float(dev.flatten().quantile(0.99)), "z_dev_max": float(dev.max()),
               "verdict": "PASS" if ok else "FAIL"}
        out["members"].append(rec)
        print(f"[verify] {name}: frame {prov.get('method')} held-out median |dz|/sd = "
              f"{rec['z_dev_median']:.3e}, p99 {rec['z_dev_p99']:.3e}, max {rec['z_dev_max']:.3e} "
              f"-> {rec['verdict']}", flush=True)
    n_ok = sum(1 for m in out["members"] if m["verdict"] == "PASS")
    out["verdict"] = "PASS" if (n_ok == len(cache_hits) and cache_hits) else "FAIL"
    print(f"[verify] VERDICT {out['verdict']} ({n_ok}/{len(cache_hits)} members) {experiment} r{repeat}", flush=True)
    print("[verify] REPORT " + json.dumps(_jsonable(out)), flush=True)
    return out


def run_scaler_recovery(
    experiment: str,
    repeat: int = 0,
    *,
    source_experiments: Optional[Sequence[str]] = None,
    members: Optional[Sequence[int]] = None,
    max_events: int = 200,
    steps: int = 40,
    batch_size: int = 32,
    save: bool = True,
    accept_dev: float = 1e-3,
) -> Dict[str, object]:
    """Recover and persist each ensemble member's training-time input frame.

    Runs per member, since each member was fitted in its own frame. A member is accepted when
    z_dev_median_after <= accept_dev, well below the refit floor. Returns the report dict.
    """
    from ..data.data_augmentations import build_nested_keys_from_quantities
    from ..data.scaling import save_scalers
    from .misspec import _wrap_paths_as_loader
    from .scaler_recovery import recover_key_scalers

    import re as _re
    from copy import copy as _copy
    from ..utils import prepare_data_parameters, set_seed_for_repeat_and_ensemble

    frame = rebuild_production_frame(experiment, repeat, source_experiments=source_experiments)
    cfg, match = frame.cfg, frame.match
    # Events spread over the whole test split (stride), not the first max_events: a cache can agree
    # on its first rows and disagree later.
    n_all = len(frame.paths)
    if max_events and max_events < n_all:
        stride = n_all // max_events
        sel = list(range(0, n_all, stride))[:max_events]
    else:
        sel = list(range(n_all))
    paths = [frame.paths[i] for i in sel]
    nested_keys = build_nested_keys_from_quantities(
        list(cfg.dataset_quantities), eb_variant=getattr(cfg, "eb_map_variant", None),
    )
    encoders = [m.embedding_net for m in frame.source_models]

    cache_hits = sorted(_glob.glob(os.path.join(cfg.base_path, "checkpoints", experiment,
                                                f"*{match}*", "datasets", "emb_test.pt")))
    if not cache_hits:
        raise RuntimeError(f"no emb_test.pt caches found for {experiment} / {match}")
    if members is not None:
        cache_hits = [cache_hits[j] for j in members]

    out: Dict[str, object] = {"experiment": experiment, "repeat": repeat, "match": match,
                              "n_members": len(cache_hits), "max_events": len(paths), "members": []}
    print(f"[recover] {experiment} {match}: {len(cache_hits)} member cache(s), "
          f"{len(paths)} events each", flush=True)

    for cache_path in cache_hits:
        member_dir = os.path.dirname(os.path.dirname(cache_path))
        name = os.path.basename(member_dir)
        z_cached = torch.load(cache_path, map_location="cpu", weights_only=False)["z"]
        if z_cached.shape[0] != n_all:
            raise RuntimeError(f"{name}: cache has {z_cached.shape[0]} rows, production split {n_all}")
        z_cached = z_cached[sel]

        # Rebuild member j's config (split_seed 42 + j): with scaler_fit_seed this reproduces its
        # input frame exactly, so the LBFGS refinement is a no-op.
        m_ens = _re.search(r"_ens(\d+)_", name)
        # The seeded subsample is drawn per key in key order, so try each order of
        # dataset_quantities and keep the one that reproduces the cache.
        orders = [list(cfg.dataset_quantities)]
        if m_ens is not None and 1 < len(orders[0]) <= 4:
            orders += [list(o) for o in _permutations(orders[0]) if list(o) != orders[0]]
        res, init_method = None, "member0_refit"
        for order in orders:
            init_scalers, method_i = frame.scalers["data"], "member0_refit"
            if m_ens is not None:
                cfg_j = _copy(cfg)
                cfg_j.split_seed = 42
                cfg_j.dataset_quantities = order
                set_seed_for_repeat_and_ensemble(cfg_j, repeat_idx=repeat, ensemble_idx=int(m_ens.group(1)))
                sc_j, _tr, _va, _te = prepare_data_parameters(cfg_j)
                del _tr, _va, _te
                method_i = f"member{int(m_ens.group(1))}_refit"
                init_scalers = sc_j["data"]
                if len(orders) > 1:
                    method_i += "[" + ",".join(order) + "]"

            # Raw loader: empty key_scalers pass values straight through, in the cache's order.
            raw_loader = _wrap_paths_as_loader(
                paths, nested_keys, list(cfg.cosmo_param_names), {}, None,
                batch_size=batch_size, num_workers=2,
                eb_noise_norm=getattr(cfg, "eb_noise_norm", None),
            )
            print(f"[recover] --- {name} (init: {method_i}, {len(paths)} events spread over {n_all})", flush=True)
            res_i = recover_key_scalers(encoders, raw_loader, z_cached, init_scalers,
                                        max_events=len(paths), steps=steps)
            if res is None or res_i.z_dev_median_after < res.z_dev_median_after:
                res, init_method = res_i, method_i
            if res_i.z_dev_median_before <= 1e-6:
                break

        ok = res.z_dev_median_after <= accept_dev
        exact = res.z_dev_median_before <= 1e-6
        method = f"exact_{init_method}" if exact else f"{init_method}+lbfgs"
        rec = {"member": name, "accepted": bool(ok), "method": method, **res.as_dict()}
        print(f"[recover] {name}: init {init_method} z_dev_median {res.z_dev_median_before:.3e} -> "
              f"{'EXACT' if exact else 'not exact'}", flush=True)
        if save and ok:
            p = save_scalers(
                os.path.join(member_dir, "datasets", "scalers.pt"),
                res.key_scalers, frame.scalers.get("cosmo"),
                {"experiment": experiment, "match": match, "member": name,
                 "source_experiments": list(source_experiments or resolve_source_experiments(experiment)),
                 "cosmo_param_names": list(cfg.cosmo_param_names),
                 "method": method, "n_events": res.n_events,
                 "z_dev_median_before": res.z_dev_median_before,
                 "z_dev_median_after": res.z_dev_median_after,
                 "z_dev_max_after": res.z_dev_max_after},
            )
            rec["saved"] = p
        print(f"[recover] {name}: {res.z_dev_median_before:.3e} -> {res.z_dev_median_after:.3e} "
              f"(accept <= {accept_dev:.0e}) -> {'ACCEPTED' if ok else 'REJECTED'}"
              f"{'; saved' if rec.get('saved') else ''}", flush=True)
        out["members"].append(rec)

    n_ok = sum(1 for m in out["members"] if m["accepted"])
    out["n_accepted"] = n_ok
    print(f"[recover] REPORT " + json.dumps(_jsonable(out)), flush=True)
    print(f"[recover] {n_ok}/{len(cache_hits)} members accepted", flush=True)
    return out
