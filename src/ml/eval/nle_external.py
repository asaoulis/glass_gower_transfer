"""Score a trained NLE ensemble on data it was not trained on — including a single observation.

WHAT THIS IS FOR
----------------
The Stage-B NLE rows (`gower_nle_finetune_*`) bundle their evaluation into the training job: the
model is scored on the TEST split of its own `data_patterns`, with scalers fit on its own train
split. That is the right thing during a campaign and the wrong thing for everything after it. Two
jobs need the same model pointed at *different* files:

  * the matched-nuisance variate comparison — 160 mocks per arm at 10 near-fiducial cosmologies;
  * ⭐ **the real-data analysis** — ONE KiDS observation.

Both are the same operation, so they are one code path. The design invariant that makes that
possible:

    **The mock HDF5 schema IS the observation contract.**

An observation is a file with the same `cls_results/` + `pixelised_results/` groups a mock has and
*no* `cosmo_dict`. `H5CosmoDataset(allow_missing_cosmo_params=True)` already tolerates that, so the
real-data run is `paths=[obs.h5]` — not a new pipeline.

THE HAZARD THIS EXISTS TO AVOID
-------------------------------
Repointing a config's `data_patterns` at a small external store and re-running the normal eval is
silently wrong in three separate ways, none of which raises:

  1. `split_by_cosmology` over 10 cosmologies hits its no-train/val fallback;
  2. the bandpower/map scalers are **refit** on those 10 cosmologies (or on one observation), so the
     model sees inputs in a different frame from the one it was trained in;
  3. `build_embedding_dataloaders` keys its `emb_*.pt` cache on the RUN, not on the data, so with
     `use_cache_if_exists=True` it can return embeddings computed from an entirely different
     dataset.

This module instead fits the scalers ONCE on the experiment's own `data_patterns` and injects them,
builds the loader from an explicit path list with no split, loads (never fits) the Stage-A whitener,
and bypasses the embedding cache. Everything else — source-encoder resolution, ensemble member
discovery, checkpoint selection — is the production path in `embeddings/train.py`, threaded rather
than forked, so it cannot drift.

OUTPUTS
-------
Written under `{base_path}/checkpoints/<exp>/external/<tag>/`, never to the `ensemble_*_<match>`
names the production eval owns — so an external run can never overwrite a campaign result.
`<tag>` names the dataset (`gower_match_nla_m`, `kids_legacy_dr5`, …) and is decoupled from the glob
that resolved it.

The prior is part of the identity of an NLE posterior, so it is a first-class argument and appears
in the filename: `external_posterior_samples_<prior>_<match>.npz`.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# How many times the MEASURED scaler-refit noise floor a reproduction's z-deviation may reach
# before it stops being attributable to the refit. 3x is the band the gate has always printed its
# verdict against; it is now also the pass criterion (see run_reproduction_check CHECK 3).
_Z_FLOOR_BAND = 3.0

_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"


# --------------------------------------------------------------------------------------------
# Priors. Hoisted out of `gen_samples.py` so the external path and the sample driver cannot
# disagree about what "the kids_s8_analytic prior" means — the real-data run needs the KiDS
# analytic S8 prior, not the Gower empirical one.
# --------------------------------------------------------------------------------------------
PRIOR_MODES = ("gower", "kids_s8_analytic", "LCDM_fixed_w0")

# prior mode -> the parameters it PINS, {name: physical value}. Empty for a mode that samples the
# full vector. This table is the single source of truth for two things that must never disagree:
# what `build_prior_for_mode` pins at sampling time, and what `free_param_names` tells a READER to
# expect in the dump (see the warning on that function).
FIXED_BY_PRIOR_MODE = {
    "gower": {},
    "kids_s8_analytic": {},
    "LCDM_fixed_w0": {"w0": -1.0},
}


def free_param_names(prior_mode: str, param_names):
    """-> the parameters actually SAMPLED under `prior_mode`, in dump-column order.

    ⚠️ READ THIS BEFORE UNSCALING A DUMP. When a prior mode pins a parameter, sbi's
    `conditional_potential` samples only `dims_to_sample`, so the saved ``samples`` array has one
    column PER FREE PARAMETER — the pinned column is absent, not constant. ``theta0s`` is written
    by the loader and keeps the FULL vector. So a dump from `LCDM_fixed_w0` has
    ``samples.shape[-1] == len(param_names) - 1`` while ``theta0s.shape[-1] == len(param_names)``,
    and unscaling ``samples`` with the full min/max box is a shape error at best and a silent
    column shift at worst.

    Readers should DROP the pinned names rather than re-insert the pinned value: a constant column
    has zero variance, which turns every standardisation into NaN and breaks KDE-based corner
    plots. Everything downstream aligns parameters by NAME, so dropping composes cleanly.
    """
    fixed = FIXED_BY_PRIOR_MODE.get(prior_mode, {})
    return [n for n in param_names if n not in fixed]


def build_prior_for_mode(prior_mode: str, param_names, *, preset_overrides=None):
    """-> (prior, fixed_parameters). `prior_mode` is one of PRIOR_MODES.

    When `fixed_parameters` is not None the resulting dump is missing those columns — see
    `free_param_names`.
    """
    from ..embeddings.embeddings_utils import COSMO_PARAM_PRESET_MINMAX, _build_cosmo_preset_scaler
    from .utils import build_gower_prior, build_s8_analytic_prior

    param_names = list(param_names)
    if prior_mode == "gower":
        kw = {"preset_overrides": preset_overrides} if preset_overrides else {}
        return build_gower_prior(param_names, **kw), None
    if prior_mode == "kids_s8_analytic":
        scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        return build_s8_analytic_prior(param_names, scaler), None
    if prior_mode == "LCDM_fixed_w0":
        from gen_samples import _build_fixed_parameters_list
        scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, param_names)
        prior = build_s8_analytic_prior(param_names, scaler, return_restricted=False)
        return prior, _build_fixed_parameters_list(dict(FIXED_BY_PRIOR_MODE[prior_mode]),
                                                   param_names, space="physical")
    raise ValueError(f"unknown prior mode {prior_mode!r}; choose from {PRIOR_MODES}")


# --------------------------------------------------------------------------------------------
# Datasets addressable by a bare name (the gatekeeper --args charset forbids '/', '*' and '=',
# so a cluster caller can only pass names; the name -> glob mapping has to live in code).
# --------------------------------------------------------------------------------------------
def store_glob(name: str) -> str:
    """A store basename -> its output glob: gpu5 first, then the gpu4 datasets root (the matched
    near-fiducial stores `gower_match_*_bgp` live there)."""
    import os as _os
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
    scalers: Dict[str, object]          # the ORIGINAL training scalers ('data' / 'cosmo')
    test_loader: object                 # embedding loader over the external paths
    raw_dataset: object = None          # the H5CosmoDataset behind it (for path/id bookkeeping)
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
        },
    )


def match_string_for(experiment: str, repeat: int) -> str:
    """Repeat index -> the run match string for a Stage-B row.

    NOT constructible from a single rule: the `nla_m` flagship rows were trained with
    `match_num_cosmo` on and carry `ncosmo300_<r>`, while every later chain carries
    `ncosmoNone_<r>`. Getting this wrong resolves NO checkpoint (loud) or, worse, a different
    repeat's (silent), so the mapping is read off the config rather than guessed.
    """
    from config.experiments import experiments as _exps
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments as _bgp
    merged = {**_exps, **_bgp}
    exp = merged.get(experiment, {})
    n = exp.get("max_trainval_cosmos", None)
    if isinstance(n, (list, tuple)):
        n = n[0] if len(n) == 1 else None
    tag = "ncosmoNone" if n is None else f"ncosmo{int(n)}"
    return f"{tag}_{int(repeat)}"


def resolve_source_experiments(experiment: str) -> Sequence[str]:
    """The frozen source encoder(s) for a Stage-B row, from the config side (one place)."""
    from config.kids_legacy_bgp import HF_RETRAIN_SOURCES, FLAGSHIP_NLE_SOURCES
    for table in (HF_RETRAIN_SOURCES, FLAGSHIP_NLE_SOURCES):
        if experiment in table:
            src = table[experiment]
            return [src] if isinstance(src, str) else list(src)
    raise KeyError(
        f"no source encoder registered for '{experiment}'. Add it to HF_RETRAIN_SOURCES or "
        f"FLAGSHIP_NLE_SOURCES in config/kids_legacy_bgp.py — the source encoder is part of a "
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
    """Run the ensemble's MCMC over every row of `p.test_loader`.

    `num_samples` is TOTAL per observation (sbi's `sample_batched` splits it across `num_chains`),
    so raising `num_chains` shortens each chain rather than multiplying the work — which is what
    makes the N=1 case parallel without any cross-process replication.

    ⭐ N=1 (the real-data run). The joblib fan-out is over EVENTS, so one observation = one work
    item = ONE core busy on a 40/64-core node. `mcmc_workers=K` splits the sample budget of each
    event across K processes instead (see `EnsembleLikelihoodNDELightningModule.generate_samples`);
    combined with `num_chains` the per-chain length is `num_samples / (K * num_chains)`, which is
    what actually moves the wall clock once `warmup_steps` stops dominating. Pooling K independent
    chain groups is identical in distribution to one process with K x num_chains chains, so the
    posterior is unchanged. `mcmc_threads=1` keeps the K processes from oversubscribing.
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
                             source_experiments=source_experiments, batch_size=batch_size)
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

    # An observation has no truth. Everything metric-shaped is scored against theta0s, so it is
    # skipped rather than silently computed against NaN.
    t = torch.as_tensor(theta0s) if not torch.is_tensor(theta0s) else theta0s
    has_truth = bool(torch.isfinite(t).all())
    if compute_metrics is None:
        compute_metrics = has_truth
    if compute_metrics and not has_truth:
        raise ValueError("compute_metrics=True but the inputs carry no finite truth vector")

    npz = os.path.join(out, f"external_posterior_samples_{prior_mode}_{p.match_string}.npz")
    _save_posterior_samples(npz, theta0s, samples, _paths_of(p, paths))
    print(f"[nle-external] wrote {npz}", flush=True)

    # HOW the posterior was drawn is part of the record, not a runtime detail: for the blind run
    # the provenance json is the only place the chain/warmup/shard geometry survives, and the
    # between-shard R-hat a reader can compute from the npz needs `mcmc_workers` to know where
    # the shard boundaries are (block s = rows [s*ceil(N/K), min((s+1)*ceil(N/K), N))).
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
                # A pinning mode samples only the free dimensions, so `samples` is D-1 wide while
                # `theta0s` is D: scoring them against the full cosmo scaler broadcast-errors and
                # (before 2026-09-08) silently lost every LCDM_fixed_w0 metrics json. Restrict both
                # truth and scaler to the free columns. The prior is dropped rather than
                # conditioned: `build_prior_for_mode` returns the FULL-dimensional object, and a
                # prior of the wrong dimension would produce plausible, wrong shrinkage numbers.
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
            # NB argument order is (theta0s, samples, param_scaler) — the scaler carries
            # `.parameter_names`, so the cosmo scaler is what run_evaluation_on_samples wants.
            metrics = run_evaluation_on_samples(
                score_theta0s, samples, score_scaler, prior=prior, compute_calibration=False,
            )
            mpath = os.path.join(out, f"external_evaluation_results_{prior_mode}_{p.match_string}.json")
            with open(mpath, "w") as fh:
                json.dump(_jsonable(metrics), fh, indent=2)
            result["metrics_json"] = mpath
            print(f"[nle-external] wrote {mpath}", flush=True)
        except Exception as exc:                       # metrics are a bonus, samples are the point
            print(f"[nle-external] metrics skipped ({type(exc).__name__}: {exc})", flush=True)

    with open(os.path.join(out, f"external_provenance_{p.match_string}.json"), "w") as fh:
        json.dump(_jsonable(result), fh, indent=2)
    return result


@dataclass(frozen=True)
class ProductionFrame:
    """A Stage-B run's TRAINING frame, rebuilt exactly as `embeddings/train.py` built it."""
    cfg: object
    match: str
    src_match: str
    source_models: list
    scalers: dict
    paths: list


def rebuild_production_frame(experiment: str, repeat: int = 0, *,
                             source_experiments: Optional[Sequence[str]] = None) -> ProductionFrame:
    """Rebuild the exact training frame of a Stage-B run: cfg, source encoders, scalers, test split.

    ONE definition, used by both the reproduction gate and scaler recovery. They must agree
    field for field -- the gate's entire value is that it mirrors production, and recovery fits a
    frame against embeddings produced by that same production path, so two drifting copies of this
    logic would make both meaningless while still looking green.

    The ORDER here is load-bearing, not cosmetic. `prepare_data_parameters` resolves the HDF5
    nested keys from `cfg.dataset_quantities`, and for a Stage-B row those RAW-DATA quantities come
    from the SOURCE ENCODER -- the Stage-B config itself describes embeddings, not maps. Production
    does exactly this (embeddings/train.py:308-340): load_pretrained_models ->
    cfg.dataset_quantities -> prepare_data_parameters. Calling it any earlier leaves
    dataset_quantities unset and `dict(config.dataset_nested_keys)` raises on the None default.

    Note `match_string` is set to the SOURCE match (None_<r>), NOT the Stage-B match, because that
    is what production's `cfg_0` uses; a cfg differing from `cfg_0` in ANY field would fit
    different scalers.
    """
    from ..embeddings.train import build_cfg_from_experiment_dict
    from ..embeddings.embeddings_utils import load_pretrained_models
    from ..utils import prepare_data_parameters, set_seed_for_repeat_and_ensemble
    from ..models.sampling import get_dataset_paths
    from config.experiments import experiments as _exps
    from config.kids_legacy_bgp import kids_legacy_bgp_experiments as _bgp

    merged = {**_exps, **_bgp}
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
    cfg.test_shape_noise_idx = [0]      # what embeddings/train.py sets; keeps 4 of 16 per cosmology
    cfg.split_seed = 42
    set_seed_for_repeat_and_ensemble(cfg, repeat_idx=repeat, ensemble_idx=0)
    scalers, _tr, _va, test_loader = prepare_data_parameters(cfg)

    # `TransformingDataset` keeps the H5 dataset on `.base_ds`, NOT `.dataset`, so a naive getattr
    # chain silently yields [] and every downstream check becomes vacuously "fine". Reuse the
    # helper that already walks both wrappers.
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
    """⭐ THE ANTI-REFIT PROOF. Point this path at the model's OWN test set and check it reproduces
    what the production eval produced.

    Construction is not evidence: every guarantee this module claims (original scalers injected, no
    split, whitener loaded not fit, cache bypassed) is invisible when it fails — a refit scaler
    produces perfectly plausible numbers in the wrong frame. The only way to know is to run the new
    path over the data the production run already scored and compare.

    Three checks, cheapest first, no sampling:

      1. **File-set equality** vs the production `ensemble_posterior_samples_<match>.npz` — plus a
         row-count cross-check. ⚠️ Usually **UNAVAILABLE**: those dumps record only
         ('samples', 'theta0s'), and the cached `emb_test.pt` serialises no paths either, so there
         is nothing to compare basenames against. It is reported as `unavailable` rather than
         skipped in silence, and CHECK 2 carries the weight instead.
      2. ⭐ **EXACT split reproduction via `theta`**, against the run's cached `emb_test.pt`. This
         is the load-bearing check. `theta` is read straight from the HDF5 files and scaled by the
         DETERMINISTIC preset min/max cosmo scaler — no stochastic subsample anywhere on that path
         — so rebuilding the same events in the same order must match bit for bit. A wrong split, a
         wrong shape-noise filter, or a wrong ordering all fail here, and it runs before the
         expensive second embedding pass. This tests exactly what CHECK 1 was specified to test,
         but by value rather than by name, and without a tolerance.
      3. **Raw `z` vs the same cached `emb_test.pt`**, which is the actual embedding the trained
         flow consumed. Unlike 2 this CANNOT be exact — see below.

    ⚠️ Check 3 cannot be exact. `_fit_data_key_scalers_from_paths` subsamples 1000 files with the
    GLOBAL, unseeded RNG, so the training-time scalers are unrecoverable for an already-trained run.
    The check therefore ALSO measures the irreducible floor by refitting twice under different RNG
    states, and requires the reproduction deviation not to exceed it materially. A deviation well
    above the floor means something other than the scaler subsample differs — stop and diagnose
    rather than widening the tolerance.
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

    # ---- CHECK 1: file-set equality vs the production posterior dump ----------------------------
    # The dump names this key `test_files`, NOT `files` (alongside `sim_ids` / `aug_ids`). Older
    # dumps -- e.g. the 2026-07-08 flagship ones -- carry only ('samples', 'theta0s') and genuinely
    # cannot support this check, so report `unavailable` explicitly rather than skipping in silence
    # (the original bug: a key-name miss took neither branch and printed nothing at all).
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
                # CHECK 1c: align on basename and compare the recorded ids row-wise. Order is NOT
                # required to match -- production order comes from split_by_cosmology, this path
                # preserves the given path order -- so align first, then compare.
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

    # ---- CHECK 3: raw z vs the cached emb_test.pt ----------------------------------------------
    # (source_models / cfg.dataset_quantities were resolved above -- see the ordering note.)

    def _raw_z(key_scalers, cosmo_scaler):
        loader, _ds = _build_external_embedding_loader_for_cfg(
            cfg, source_models, prod_paths, key_scalers, cosmo_scaler,
            whiten_cfg=None,                      # RAW z: the cache stores pre-whitening embeddings
            batch_size=batch_size,
        )
        zs, ths = [], []
        for b in loader:                       # ONE pass: the embedding forward is the cost here
            zs.append(b[0])
            ths.append(b[1])
        return _torch.cat(zs, dim=0), _torch.cat(ths, dim=0)

    z_ours, th_ours = _raw_z(scalers_a["data"], scalers_a["cosmo"])

    # Locate the run's cached embeddings. An ensemble writes one per member
    # (pretrain_<match>_ens<j>_<source>/datasets/emb_test.pt); they share the split and differ only
    # by each member's own stochastic scaler fit, so compare against ONE and say which.
    cache_hits = sorted(_glob.glob(os.path.join(cfg.base_path, "checkpoints", experiment,
                                                f"*{match}*", "datasets", "emb_test.pt")))
    report["n_cache_candidates"] = len(cache_hits)
    cached = _torch.load(cache_hits[0], map_location="cpu") if cache_hits else None
    if cached is not None:
        report["cache_used"] = cache_hits[0]
        print(f"[repro] cache: {len(cache_hits)} member caches found; comparing against "
              f"{os.path.basename(os.path.dirname(os.path.dirname(cache_hits[0])))}", flush=True)

    # ---- CHECK 2: EXACT split reproduction, via theta ------------------------------------------
    # This is what actually carries the weight (CHECK 1 cannot run -- no paths are recorded
    # anywhere). theta is read straight from the HDF5 files and scaled by the DETERMINISTIC preset
    # min/max cosmo scaler -- no stochastic subsample anywhere in that path -- so if we rebuilt the
    # same events in the same order it must match the cache BIT FOR BIT. A different split, a
    # different shape-noise filter, or a different ordering all show up here immediately, before
    # the expensive second embedding pass.
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

    # The irreducible floor: refit the scalers on a DIFFERENT 1000-file subsample and re-embed.
    # This must vary `scaler_fit_seed`, not the global RNG: the fit no longer consumes the global
    # stream (it takes a local Generator seeded from the config), so re-seeding numpy would leave
    # the subsample identical and collapse the measured floor to exactly zero -- silently turning
    # CHECK 3 into a comparison against nothing. The quantity we want is unchanged either way:
    # how far does z move when the subsample choice moves?
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
            # The criterion is FLOOR-RELATIVE by design (plan 3.0.8), not a fixed absolute number.
            # The training-time scalers are unrecoverable (stochastic 1000-file subsample), so the
            # question is never "is the deviation small in absolute terms" but "is it bigger than
            # the irreducible refit noise we just measured". A fixed tolerance sitting BELOW the
            # measured floor can never pass: the first clean run measured floor = 1.11e-2 against a
            # hardcoded tol of 1e-2 and duly reported FAIL at 1.29e-2 -- a deviation just 1.16x the
            # floor, i.e. exactly what a correct reproduction looks like. z_tol is kept as an
            # absolute LOWER BOUND on the budget, so a tiny floor still demands a tiny deviation.
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
            # Plan risk #7: a floor above z_tol is itself a finding -- the SAME irreducible
            # uncertainty attaches to the production misspec numbers, which came through this very
            # refit path. Surface it rather than burying it in a PASS.
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

    # ---- CHECK 4: the RECOVERED frame, if one has been persisted --------------------------------
    # The generalisation test for scaler recovery. The fit only ever saw a few hundred events; this
    # scores the recovered frame over the FULL split, so a frame that merely memorised its fitting
    # subset shows up here as a deviation no better than the refit's.
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
                # Read the comparison numbers from the report, not from CHECK 3's locals:
                # `dev`/`floor_med` only exist on that check's success branch.
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
    """Recover and persist each ensemble member's TRAINING-TIME input frame.

    Runs PER MEMBER on purpose. The nine members were each fitted on their own stochastic
    1000-file subsample, so they consumed nine different frames -- measured directly from their
    caches (z: ens0-ens1 1.205e-2, ens1-ens2 1.102e-2, while theta is identical to 0.000e+00).
    One recovered frame would therefore be right for one member and ~1e-2 wrong for the other
    eight, which is the very error this exists to remove.

    Acceptance is `z_dev_median_after <= accept_dev` (default 1e-3), an order of magnitude BELOW
    the ~1.1e-2 refit floor. Landing merely "under budget" at ~floor would mean the fit did not
    find the training draw, and is reported as a FAIL rather than quietly accepted.
    """
    from ..data.data_augmentations import build_nested_keys_from_quantities
    from ..data.scaling import save_scalers
    from .misspec import _wrap_paths_as_loader
    from .scaler_recovery import recover_key_scalers

    frame = rebuild_production_frame(experiment, repeat, source_experiments=source_experiments)
    cfg, match = frame.cfg, frame.match
    paths = frame.paths[:max_events] if max_events else frame.paths
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

        # RAW loader: empty key_scalers => DataDictScalerTransform passes values straight through.
        # Order must match the cache, which the reproduction gate proves (CHECK 2, theta exact).
        raw_loader, _ = _wrap_paths_as_loader(
            paths, nested_keys, list(cfg.cosmo_param_names), {}, None,
            batch_size=batch_size, num_workers=2,
            eb_noise_norm=getattr(cfg, "eb_noise_norm", None),
        )
        print(f"[recover] --- {name}", flush=True)
        # NB frame.scalers is {'data': {...}, 'cosmo': ...}; recovery wants the KEY scalers.
        res = recover_key_scalers(encoders, raw_loader, z_cached, frame.scalers["data"],
                                  max_events=len(paths), steps=steps)

        ok = res.z_dev_median_after <= accept_dev
        rec = {"member": name, "accepted": bool(ok), **res.as_dict()}
        if save and ok:
            p = save_scalers(
                os.path.join(member_dir, "datasets", "scalers.pt"),
                res.key_scalers, frame.scalers.get("cosmo"),
                {"experiment": experiment, "match": match, "member": name,
                 "source_experiments": list(source_experiments or resolve_source_experiments(experiment)),
                 "cosmo_param_names": list(cfg.cosmo_param_names),
                 "method": "recovered", "n_events": res.n_events,
                 "z_dev_median_after": res.z_dev_median_after},
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
