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

_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"


# --------------------------------------------------------------------------------------------
# Priors. Hoisted out of `gen_samples.py` so the external path and the sample driver cannot
# disagree about what "the kids_s8_analytic prior" means — the real-data run needs the KiDS
# analytic S8 prior, not the Gower empirical one.
# --------------------------------------------------------------------------------------------
PRIOR_MODES = ("gower", "kids_s8_analytic", "LCDM_fixed_w0")


def build_prior_for_mode(prior_mode: str, param_names, *, preset_overrides=None):
    """-> (prior, fixed_parameters). `prior_mode` is one of PRIOR_MODES."""
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
        return prior, _build_fixed_parameters_list({"w0": -1.0}, param_names, space="physical")
    raise ValueError(f"unknown prior mode {prior_mode!r}; choose from {PRIOR_MODES}")


# --------------------------------------------------------------------------------------------
# Datasets addressable by a bare name (the gatekeeper --args charset forbids '/', '*' and '=',
# so a cluster caller can only pass names; the name -> glob mapping has to live in code).
# --------------------------------------------------------------------------------------------
def store_glob(name: str) -> str:
    """A gpu5 store basename -> its output glob."""
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
):
    """Run the ensemble's MCMC over every row of `p.test_loader`.

    `num_samples` is TOTAL per observation (sbi's `sample_batched` splits it across `num_chains`),
    so raising `num_chains` shortens each chain rather than multiplying the work — which is what
    makes the N=1 case parallel without any cross-process replication.
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

    result = {"output_dir": out, "samples_npz": npz, "n_files": len(paths),
              "has_truth": has_truth, "prior": prior_mode,
              "experiment": experiment, "match_string": p.match_string,
              "provenance": p.provenance}

    if compute_metrics:
        from .evaluate_models import run_evaluation_on_samples
        try:
            prior, _fx = build_prior_for_mode(prior_mode, p.param_names)
            # NB argument order is (theta0s, samples, param_scaler) — the scaler carries
            # `.parameter_names`, so the cosmo scaler is what run_evaluation_on_samples wants.
            metrics = run_evaluation_on_samples(
                theta0s, samples, p.scalers["cosmo"], prior=prior, compute_calibration=False,
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
