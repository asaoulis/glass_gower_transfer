"""Model-misspecification evaluation: ONE trained NPE ensemble evaluated on the TEST split of
MANY Gower variate datasets, reusing the ORIGINAL training scalers.

Motivation: the production NPE (`gower_npe_finetune_nla_m_z8`, repeat 0) was trained on the
nla_m Gower suite. Each physics variate (nla, nla_z, galaxy-bias, ...) is a controlled
misspecification of that forward model; running the SAME model on each variate's held-out test
cosmologies and measuring the TARP miscalibration quantifies how sensitive the inference is to
that misspecification.

Key differences from the standard `evaluate_best_checkpoint` path:
- Data scalers (bandpowers LogNormal + map Standard) are fit ONCE on the original nla_m
  train+val split via `prepare_data_parameters` and INJECTED into every variate test loader —
  never refit on a variate (a refit would absorb part of the covariate shift being measured).
- Variate test sets are built directly from the shared fixed-test lock file (lock ∩ on-disk),
  NOT via `split_by_cosmology`: variates were largely simulated ON the 200 held-out test ids,
  so forcing them all into test would trigger split_by_cosmology's no-train/val fallback and
  silently produce a train-heavy split.
- Cosmo params absent from a variate (e.g. b_ia in nla/nla_z) are NaN-filled by the loader
  (`allow_missing_cosmo_params`), keeping theta 9-dim to match the flow; calibration is then
  computed only over the finite dims. Per-variate `exclude_params` additionally drops params
  whose MEANING differs between suites (a_ia under NLA vs NLA-M parametrisations).
- FoM is still computed over all 9 sampled dims against the base Gower prior.

Invoked from eval.py (RUN_MISSPEC flag) so the cluster's `eval-submit` job needs no new verb.
"""
import json
import os
import traceback
from copy import copy
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

from ..data.data_loaders import H5CosmoDataset, build_nested_keys_from_quantities
from ..data.data_selection import (
    _filter_paths_by_shape_noise_idx,
    collect_paths,
    extract_cosmo_index,
)
from ..data.fixed_test_set import resolve_fixed_test_ids
from ..utils import (
    DataDictScalerTransform,
    TransformingDataset,
    build_ensemble_model_from_checkpoints,
    prepare_data_parameters,
)
from .evaluate_models import (
    DimNormalizedFoMDiagnostics,
    StandardFoMDiagnostics,
    TARPDiagnostics,
    _sample_from_prior,
    _split_samples_first_and_sims_first,
    rescale_parameters,
)
from .utils import (
    _config_preset_overrides,
    _pop_credible_intervals,
    _save_posterior_samples,
    _to_json_compatible,
    build_gower_prior,
)

_GPU5 = "/share/gpu5/asaoulis/transfer_datasets"
_GPU4 = "/share/gpu4/asaoulis/transfer_datasets"

# a_ia is EXCLUDED from calibration for variates whose IA parametrisation differs from the
# nla_m training suite (its meaning changes between suites — user directive 2026-07-08).
# nla_m / gb* share the nla_m IA model, so a_ia keeps its meaning there.
DEFAULT_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_counts/output_*.h5", "exclude_params": []},
]

# GLASS pre-train variate suites (gpu5 f16 fwhm4_lmin56_lcut1400 prebakes, matching the GLASS
# z8 foundation's maps). In-dist = the foundation's own lmin50 training store; its test ids
# are derived from the training split at runtime (no fixed lock for GLASS).
GLASS_PRETRAIN_VARIATES: List[Dict] = [
    {"name": "glass_nla_m",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_lmin50_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "glass_novd",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_novd_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": []},
    {"name": "glass_nla",
     "patterns": f"{_GPU5}/glass_mocks_nla_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "glass_nla_z",
     "patterns": f"{_GPU5}/glass_mocks_nla_z_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": ["a_ia"]},
]

# --- NO-VD suite (2026-07-29 variate switch; the MAIN analysis variate) -------------------------
# Added ALONGSIDE the VD-on sets above rather than repointing them: the counts-era misspec bases
# (e.g. gower_npe_finetune_nla_m_counts_z8) still resolve, and repointing in place would have made
# any such re-run silently evaluate a VD-on model against VD-off data — different forward physics,
# no error raised. Select these with `--variates gower_novd` / `glass_pretrain_novd`.
#
# Store provenance (datasets_checklist.md): S1 nla_m + G1 glass nla_m are consumed from the gpu5 f16
# fwhm4 PREBAKES; the misspec test sets (S2/S3 nla,nla_z and S4/S5 + G2/G3 gb0p5,gb1p5) are consumed
# RAW off gpu4 — bake optional, same as the VD-on sets do.
NOVD_GOWER_VARIATES: List[Dict] = [
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z_novd_counts/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5_novd_counts/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5_novd_counts/output_*.h5", "exclude_params": []},
]

# Foundation-level (pre-Gower-finetune) misspec check — user 2026-07-29: run the gb0p5/gb1p5 GLASS
# sets against the 5 pre-trained foundations, i.e. BEFORE any Gower finetune. This is why G2/G3
# exist and launched early. NB the VD-on GLASS set above carries no gb variates; this one does.
NOVD_GLASS_PRETRAIN_VARIATES: List[Dict] = [
    {"name": "glass_nla_m",
     "patterns": f"{_GPU5}/glass_mocks_nla_m_novd_counts_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "glass_gb0p5", "patterns": f"{_GPU4}/glass_mocks_gb0p5_novd_counts/output_*.h5",
     "exclude_params": []},
    {"name": "glass_gb1p5", "patterns": f"{_GPU4}/glass_mocks_gb1p5_novd_counts/output_*.h5",
     "exclude_params": []},
    {"name": "glass_nla", "patterns": f"{_GPU4}/glass_mocks_nla_novd_counts/output_*.h5",
     "exclude_params": ["a_ia"]},
    {"name": "glass_nla_z", "patterns": f"{_GPU4}/glass_mocks_nla_z_novd_counts/output_*.h5",
     "exclude_params": ["a_ia"]},
]

VARIATE_SETS: Dict[str, List[Dict]] = {
    "gower": DEFAULT_VARIATES,
    "glass_pretrain": GLASS_PRETRAIN_VARIATES,
    "gower_novd": NOVD_GOWER_VARIATES,
    "glass_pretrain_novd": NOVD_GLASS_PRETRAIN_VARIATES,
}


def _load_experiment_config(experiment_name: str):
    """Rebuild the experiment config exactly like eval.py's load_config + list-branch handling."""
    from config.default import get_default_config
    from config.experiments import experiments as base_experiments
    from config.ablations import ablation_experiments
    from config.kids_legacy import kids_legacy_experiments
    from config.kids_legacy_counts import kids_legacy_counts_experiments
    from config.kids_legacy_novd import kids_legacy_novd_experiments

    exps = dict(base_experiments)
    exps.update(ablation_experiments)
    exps.update(kids_legacy_experiments)
    exps.update(kids_legacy_counts_experiments)
    exps.update(kids_legacy_novd_experiments)
    experiment_config = exps[experiment_name]

    config = get_default_config()
    config.experiment_name = experiment_name
    for key, val in experiment_config.items():
        if key == "max_trainval_cosmos":
            continue
        setattr(config, key, val)

    max_tv = experiment_config.get("max_trainval_cosmos", None)
    if isinstance(max_tv, (list, tuple)):
        if len(max_tv) != 1:
            print(f"[misspec] WARNING: max_trainval_cosmos sweep {max_tv}; using first entry.")
        config.max_trainval_cosmos = int(max_tv[0])
        config.match_num_cosmo = True  # match eval.py: match_string includes ncosmo
    elif max_tv is not None:
        config.max_trainval_cosmos = int(max_tv)
    return config


def _probe_variate_file(paths: Sequence[str], nested_keys: Dict, cosmo_param_names: Sequence[str]):
    """Open the first readable file and report (missing_data_keys, present_cosmo_params).

    Fails fast with a useful message (e.g. wrong eb variant tag) instead of letting the
    loader skip every file one by one.
    """
    last_err = None
    for p in paths[:16]:
        try:
            with h5py.File(p, "r") as f:
                missing = []
                for out_key, path in nested_keys.items():
                    node = f
                    for key in path:
                        if key not in node:
                            missing.append((out_key, "/".join(path)))
                            node = None
                            break
                        node = node[key]
                grp = f["cosmo_dict"]
                present_cosmo = [c for c in cosmo_param_names if c in grp]
            return missing, present_cosmo, p
        except OSError as e:  # truncated/corrupt file — try the next one
            last_err = e
            continue
    raise RuntimeError(f"No readable file among first {min(16, len(paths))} probe paths "
                       f"(last error: {last_err})")


def build_variate_test_loader(
    patterns,
    nested_keys: Dict,
    cosmo_param_names: Sequence[str],
    key_scalers: Dict,
    cosmo_scaler,
    test_id_pool,
    test_shape_noise_idx=(0, (0, 1)),
    batch_size: int = 64,
    num_workers: int = 4,
    max_test_files: Optional[int] = None,
):
    """Variate TEST loader with the ORIGINAL scalers injected (never refit).

    Test cosmologies = (``test_id_pool`` ∩ on-disk ids); falls back to ALL on-disk
    cosmologies when there is no overlap (e.g. a small gb subset outside the pool).
    ``max_test_files`` caps the test set by accumulating whole cosmologies (sorted by
    sim_id) until the file budget is reached.
    """
    all_paths = collect_paths(patterns)
    by_cosmo: Dict[int, List[str]] = {}
    for p in all_paths:
        by_cosmo.setdefault(extract_cosmo_index(p), []).append(p)
    for c in by_cosmo:
        by_cosmo[c].sort()

    pool = set(test_id_pool or [])
    test_ids = sorted(set(by_cosmo.keys()) & pool)
    used_fallback = False
    if not test_ids:
        test_ids = sorted(by_cosmo.keys())
        used_fallback = True
        print(f"[misspec] WARNING: no overlap with the {len(pool)} test-pool ids; "
              f"falling back to ALL {len(test_ids)} on-disk cosmologies (model may have seen "
              "these cosmologies at nla_m physics).", flush=True)

    test_paths = [p for c in test_ids for p in by_cosmo[c]]
    filtered = _filter_paths_by_shape_noise_idx(test_paths, list(test_shape_noise_idx))
    if not filtered:
        print(f"[misspec] WARNING: shape-noise filter {test_shape_noise_idx} matched no files; "
              "using all test-cosmology files.", flush=True)
        filtered = test_paths

    if max_test_files is not None and len(filtered) > int(max_test_files):
        by_id: Dict[int, List[str]] = {}
        for p in filtered:
            by_id.setdefault(extract_cosmo_index(p), []).append(p)
        capped, kept_ids = [], []
        for c in sorted(by_id):
            if capped and len(capped) + len(by_id[c]) > int(max_test_files):
                break
            capped.extend(by_id[c])
            kept_ids.append(c)
        print(f"[misspec] max_test_files={max_test_files}: capped {len(filtered)} -> "
              f"{len(capped)} files ({len(kept_ids)}/{len(test_ids)} cosmologies, "
              "first by sorted sim_id).", flush=True)
        filtered = capped
        test_ids = kept_ids

    ds = H5CosmoDataset(
        filtered,
        nested_keys,
        list(cosmo_param_names),
        transform=None,
        allow_missing_cosmo_params=True,
    )
    wrapped = TransformingDataset(
        ds,
        data_transform=DataDictScalerTransform(key_scalers),
        cosmo_scaler=cosmo_scaler,
    )
    loader = torch.utils.data.DataLoader(
        wrapped, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    meta = {
        "n_test_files": len(filtered),
        "n_test_cosmologies": len(test_ids),
        "test_ids_from_fixed_lock": not used_fallback,
        "test_paths": filtered,
    }
    return loader, meta


def _compute_misspec_metrics(
    theta0s: torch.Tensor,
    samples: torch.Tensor,
    param_names: Sequence[str],
    available_idx: Sequence[int],
    param_scaler,
    prior_samples_scaled: Optional[torch.Tensor],
):
    """Calibration on the available dims + FoM vs the base prior on all sampled dims.

    theta0s ~ [N, D] and samples ~ [S, N, D], both in SCALED space (loader/flow space) —
    mirrors run_evaluation_on_samples, which runs TARP in scaled space and FoM in both.
    """
    param_names = list(param_names)
    available_idx = list(available_idx)
    available_names = [param_names[i] for i in available_idx]
    n_sims = theta0s.shape[0]

    _, samples_sims_first = _split_samples_first_and_sims_first(samples, n_sims, name="samples")
    scaled_theta0s = rescale_parameters(theta0s, param_scaler)          # physical units
    scaled_samples = rescale_parameters(samples, param_scaler)
    scaled_samples_first, scaled_samples_sims_first = _split_samples_first_and_sims_first(
        scaled_samples, n_sims, name="scaled_samples"
    )

    metrics: Dict = {"available_params": available_names}

    # --- calibration (TARP) on the available dims only ------------------------------------
    tarp = TARPDiagnostics(available_names, bootstrap=True, num_bootstrap=25, seed=None)
    metrics.update(tarp.compute_all(
        samples_sims_first[:, :, available_idx].contiguous(),
        theta0s[:, available_idx].contiguous(),
    ))

    # --- FoM vs the base Gower prior over ALL sampled dims --------------------------------
    prior_samples_unscaled = (
        rescale_parameters(prior_samples_scaled, param_scaler)
        if prior_samples_scaled is not None else None
    )
    metrics.update(DimNormalizedFoMDiagnostics(
        param_names, prior_samples_t=prior_samples_scaled,
    ).compute_all(samples_sims_first))
    metrics.update(StandardFoMDiagnostics(
        param_names, prior_samples_t=prior_samples_unscaled,
    ).compute_all(scaled_samples_sims_first))

    # --- per-available-param point stats in physical units --------------------------------
    sample_means = scaled_samples_first.mean(axis=0)
    bias = sample_means - scaled_theta0s
    std_devs = scaled_samples_first.std(axis=0)
    width_68 = (torch.quantile(scaled_samples_first, 0.84, dim=0)
                - torch.quantile(scaled_samples_first, 0.16, dim=0))
    for dim in available_idx:
        name = param_names[dim]
        metrics[name] = {
            "mse": ((bias[:, dim] ** 2).mean()).item(),
            "bias": bias[:, dim].mean().item(),
            "std_dev": std_devs[:, dim].mean().item(),
            "width_68": width_68[:, dim].mean().item(),
        }
    return metrics


def run_misspecification_eval(
    base_experiment: str = "gower_npe_finetune_nla_m_z8",
    repeat_index: Optional[int] = None,
    variates: Optional[List[Dict]] = None,
    num_samples: int = 10000,
    prior_num_samples: int = 20_000,
    test_shape_noise_idx=(0, (0, 1)),
    out_subdir: str = "misspec",
    repeat_indices: Sequence[int] = (0,),
    variate_set: Optional[str] = None,
    max_test_files: Optional[int] = None,
):
    """Evaluate the base experiment's model(s) on every variate, per training repeat.

    Works for eval-time ensembles (ensemble_repeats>1: the repeat's N members are loaded as
    one EnsembleNDELightningModule) AND single-model experiments (the repeat's best checkpoint).

    ``repeat_indices``: one full pass (scalers + model + all variates) per repeat. With >1
    repeat, per-event CROSS-REPEAT posterior disagreement (mean pairwise symmetric
    diag-Gaussian KL, as in ensemble_uncertainty.py) is computed per variate and saved next to
    the calibration results — the OOD statistic to correlate with miscalibration.
    ``repeat_index`` (int) is a legacy alias for a single-repeat run.

    Variate test cosmologies come from ``config.fixed_test_sim_ids`` when the experiment uses a
    lock file, else from the experiment's own held-out test split (derived at runtime), so no
    variate is ever evaluated on cosmologies the model trained on. ``max_test_files`` caps each
    variate's test set (whole cosmologies, sorted by sim_id).
    """
    from ..models.utils import apply_repeat_config
    from .utils import _resolve_test_paths, load_best_model_and_build_posterior

    if variates is None:
        variates = VARIATE_SETS[variate_set] if variate_set else DEFAULT_VARIATES
    if repeat_index is not None:
        repeat_indices = (int(repeat_index),)
    repeat_indices = [int(r) for r in repeat_indices]

    cfg0 = _load_experiment_config(base_experiment)
    param_names = list(cfg0.cosmo_param_names)
    eb_variant = getattr(cfg0, "eb_map_variant", None)
    nested_keys = build_nested_keys_from_quantities(list(cfg0.dataset_quantities), eb_variant)
    out_root = os.path.join(cfg0.base_path, "checkpoints", cfg0.experiment_name, out_subdir)

    # Base Gower prior + one shared set of prior samples (FoM shrinkage reference) — the
    # prior is repeat-independent.
    prior = build_gower_prior(param_names, preset_overrides=_config_preset_overrides(cfg0))
    prior_samples_scaled = _sample_from_prior(prior, prior_num_samples, target_dim=len(param_names))

    summary: Dict[str, Dict] = {}
    per_variate_repeat: Dict[str, Dict[str, Dict]] = {v["name"]: {} for v in variates}
    match_strings = []
    for r in repeat_indices:
        # Fresh config per repeat: apply_repeat_config mutates split_seed in place.
        cfg = _load_experiment_config(base_experiment)
        # Same test-point sub-selection for the in-distribution reference as for every
        # variate (rot0, inner noise {0,1}) so coverage curves are directly comparable.
        cfg.test_shape_noise_idx = list(test_shape_noise_idx)
        repeat_match, _ = apply_repeat_config(cfg, r)
        cfg.match_string = repeat_match
        match_strings.append(repeat_match)
        print(f"[misspec] base experiment '{base_experiment}' repeat={r} "
              f"match_string={cfg.match_string} params={param_names} eb_variant={eb_variant}",
              flush=True)

        # ORIGINAL scalers for THIS repeat: fit on its nla_m train+val split (ensemble path).
        scalers, _, _, in_dist_test_loader = prepare_data_parameters(cfg)
        orig_key_scalers = scalers["data"]
        orig_cosmo_scaler = scalers["cosmo"]
        print(f"[misspec] repeat {r}: original scalers rebuilt from {cfg.data_patterns} "
              f"(data keys: {sorted(orig_key_scalers)})", flush=True)

        # Test-id pool for the variates: the lock file when the experiment pins one, else the
        # experiment's own held-out test cosmologies (identical across repeats — the test slice
        # comes from the fixed rng(42) shuffle before the per-repeat trainval reshuffle).
        lock_spec = getattr(cfg, "fixed_test_sim_ids", None)
        if lock_spec:
            test_id_pool = set(resolve_fixed_test_ids(lock_spec) or [])
        else:
            held_out = _resolve_test_paths(in_dist_test_loader) or []
            test_id_pool = {extract_cosmo_index(p) for p in held_out}
            print(f"[misspec] repeat {r}: derived test-id pool from the training split "
                  f"({len(test_id_pool)} held-out cosmologies).", flush=True)

        # The repeat's model, built ONCE (loaders are swapped per variate): the N-member
        # eval-time ensemble when configured, else the repeat's single best checkpoint.
        n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
        if n_ens > 1:
            model = build_ensemble_model_from_checkpoints(
                cfg,
                in_dist_test_loader,
                match_string=cfg.match_string,
                member_test_loaders=[in_dist_test_loader] * n_ens,
            )
        else:
            loaded = load_best_model_and_build_posterior(
                cfg, ds_string_match=cfg.match_string, data_parameters=in_dist_test_loader,
            )
            model = loaded[0] if loaded else None
        if model is None:
            # e.g. a repeat whose training jobs haven't finished — skip it, keep the others.
            print(f"[misspec] repeat {r}: no '{base_experiment}' {cfg.match_string} checkpoints "
                  "yet — skipping this repeat.", flush=True)
            summary[f"repeat{r}"] = {"error": f"no checkpoints for {cfg.match_string}"}
            continue

        for variate in variates:
            name = variate["name"]
            key = f"{name}@r{r}" if len(repeat_indices) > 1 else name
            try:
                result = _eval_one_variate(
                    variate, model, cfg, nested_keys, param_names,
                    orig_key_scalers, orig_cosmo_scaler, prior_samples_scaled,
                    num_samples=num_samples,
                    test_shape_noise_idx=test_shape_noise_idx,
                    out_dir=os.path.join(out_root, name),
                    repeat_index=r,
                    test_id_pool=test_id_pool,
                    max_test_files=max_test_files,
                )
                per_variate_repeat[name][repeat_match] = result.pop("_per_event")
                summary[key] = result
            except Exception as e:
                print(f"[misspec] {name} (repeat {r}): FAILED — {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                summary[key] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Cross-repeat posterior disagreement per variate (the ensemble-agreement OOD statistic).
    if len(repeat_indices) > 1:
        for variate in variates:
            name = variate["name"]
            reps = per_variate_repeat[name]
            if len(reps) < 2:
                continue
            try:
                dis = _compute_repeat_disagreement(name, reps, os.path.join(out_root, name))
                for key in list(summary):
                    if key.startswith(f"{name}@"):
                        summary[key]["repeat_kl_mean"] = dis["kl_mean"]
            except Exception as e:
                print(f"[misspec] {name}: disagreement FAILED — {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()

    print("\n[misspec] ============ SUMMARY ============", flush=True)
    for key, res in summary.items():
        if "error" in res:
            print(f"[misspec] {key}: ERROR {res['error']}", flush=True)
        else:
            extra = (f" repeat_kl_mean={res['repeat_kl_mean']:.4f}"
                     if "repeat_kl_mean" in res else "")
            print(f"[misspec] {key}: n_test={res['n_test_files']} "
                  f"cal_full={res['cal_full']:.4f} cal_om_s8_w0={res['cal_om_s8_w0']:.4f}"
                  f"{extra} available={res['available_params']}", flush=True)
    return summary


def _compute_repeat_disagreement(name: str, per_repeat: Dict[str, Dict], out_dir: str):
    """Per-event posterior disagreement ACROSS training repeats for one variate.

    Aligns events by test-file basename (robust to per-repeat non-finite drops), stacks the
    per-repeat posterior moments to [K, N, D], and scores each event with the mean pairwise
    symmetric diag-Gaussian KL (ensemble_uncertainty.py formulation). Saved next to the
    per-repeat calibration JSONs so miscalibration and repeat-spread can be correlated
    directly. Moments are in SCALED space (same space as the TARP calibration)."""
    from .ensemble_discrepancies import diag_gaussian_symmetric_kl

    matches = sorted(per_repeat)
    common = set(per_repeat[matches[0]]["test_files"])
    for m in matches[1:]:
        common &= set(per_repeat[m]["test_files"])
    common = sorted(common)
    if not common:
        raise RuntimeError("no common test files across repeats")

    mu, var = [], []
    for m in matches:
        d = per_repeat[m]
        idx = {f: i for i, f in enumerate(d["test_files"])}
        sel = [idx[f] for f in common]
        mu.append(d["mu"][sel])
        var.append(d["var"][sel])
    mu = np.stack(mu, axis=0)    # [K, N, D]
    var = np.stack(var, axis=0)  # [K, N, D]

    kl = diag_gaussian_symmetric_kl(mu, var)  # [N]
    payload = {
        "variate": name,
        "repeat_match_strings": matches,
        "n_events": int(len(common)),
        "kl_mean": float(np.mean(kl)),
        "kl_median": float(np.median(kl)),
        "kl_p90": float(np.quantile(kl, 0.90)),
    }
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(matches)
    np.savez_compressed(
        os.path.join(out_dir, f"misspec_repeat_disagreement_{tag}.npz"),
        kl_score=kl, mu=mu, var=var,
        test_files=np.array(common), repeat_match_strings=np.array(matches),
    )
    with open(os.path.join(out_dir, f"misspec_repeat_disagreement_{tag}.json"), "w") as f:
        json.dump(_to_json_compatible(payload), f, indent=4)
    print(f"[misspec] {name}: repeat disagreement over {len(matches)} repeats, "
          f"kl_mean={payload['kl_mean']:.4f} kl_median={payload['kl_median']:.4f}", flush=True)
    return payload


def _eval_one_variate(
    variate: Dict,
    model,
    cfg,
    nested_keys: Dict,
    param_names: List[str],
    orig_key_scalers: Dict,
    orig_cosmo_scaler,
    prior_samples_scaled,
    *,
    num_samples: int,
    test_shape_noise_idx,
    out_dir: str,
    repeat_index: int = 0,
    test_id_pool=None,
    max_test_files: Optional[int] = None,
):
    name = variate["name"]
    exclude_params = list(variate.get("exclude_params", []))
    variate_nested_keys = nested_keys
    if variate.get("eb_variant") is not None:
        variate_nested_keys = build_nested_keys_from_quantities(
            list(cfg.dataset_quantities), variate["eb_variant"]
        )

    loader, meta = build_variate_test_loader(
        variate["patterns"],
        variate_nested_keys,
        param_names,
        orig_key_scalers,
        orig_cosmo_scaler,
        test_id_pool=test_id_pool,
        test_shape_noise_idx=test_shape_noise_idx,
        # Cap at 64: eval-mode (no_grad) encodes fit fine on a v100 at 64-128; the OOMs seen on
        # jobs 1316362/1316364 were a ZOMBIE PROCESS squatting on that GPU (~15GB), not batch
        # size — mitigate by resubmitting / splitting repeats across GPUs, not by shrinking.
        batch_size=min(64, int(getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", 64))),
        max_test_files=max_test_files,
    )

    # Fail fast on a wrong eb-variant tag / absent params rather than skip-looping every file.
    missing_keys, present_cosmo, probe_path = _probe_variate_file(
        meta["test_paths"], variate_nested_keys, param_names
    )
    if missing_keys:
        with h5py.File(probe_path, "r") as f:
            available_groups = (sorted(f["pixelised_results"].keys())
                                if "pixelised_results" in f else [])
        raise RuntimeError(
            f"data keys missing from {probe_path}: {missing_keys}; "
            f"on-disk pixelised_results groups: {available_groups} — prebake or fix eb_variant."
        )
    missing_params = [p for p in param_names if p not in present_cosmo]
    print(f"[misspec] {name}: n_test={meta['n_test_files']} "
          f"({meta['n_test_cosmologies']} cosmologies, "
          f"fixed_lock={meta['test_ids_from_fixed_lock']}) "
          f"missing_params={missing_params} exclude_params={exclude_params}", flush=True)

    # Swap the test set under the prebuilt model. Ensembles: the ensemble-level loader feeds
    # theta0s and compute_avg_log_prob; each member's loader feeds its own encode+sample pass.
    # Single models have no .members — the one loader drives everything.
    model.test_dataloader = loader
    for m in getattr(model, "members", []):
        m.test_dataloader = loader

    theta0s, samples = model.generate_samples(num_samples=num_samples)

    # Drop test points whose posterior samples came out non-finite (far-OOD conditioning can
    # degenerate the spline inverse even with the clamped discriminant). Keep the analysis on
    # the finite events and report the count — silent NaNs would poison TARP/FoM wholesale.
    test_paths = list(meta["test_paths"])
    event_ok = torch.isfinite(samples).all(dim=2).all(dim=0)  # samples [S, N, D] -> [N]
    n_bad_events = int((~event_ok).sum())
    if n_bad_events:
        print(f"[misspec] {name}: dropping {n_bad_events}/{len(event_ok)} test points with "
              "non-finite posterior samples (far-OOD sampling degeneracy).", flush=True)
        keep = event_ok.cpu().numpy().astype(bool)
        samples = samples[:, event_ok, :]
        theta0s = theta0s[event_ok, :]
        test_paths = [p for p, k in zip(test_paths, keep) if k]

    theta_np = theta0s.detach().cpu().numpy()
    finite = np.isfinite(theta_np).all(axis=0)
    available_idx = [i for i, p in enumerate(param_names)
                     if finite[i] and p not in exclude_params]
    dropped = [p for i, p in enumerate(param_names) if i not in available_idx]
    if not available_idx:
        raise RuntimeError("no finite/included cosmo params to calibrate on")

    metrics = _compute_misspec_metrics(
        theta0s, samples, param_names, available_idx, orig_cosmo_scaler, prior_samples_scaled,
    )
    # ΔMI needs log_prob at the TRUE theta — undefined when any dim is NaN.
    if bool(finite.all()):
        metrics["test_log_prob"] = model.compute_avg_log_prob()
    else:
        metrics["test_log_prob"] = None

    payload = {
        "variate": name,
        "experiment": cfg.experiment_name,
        "match_string": cfg.match_string,
        "repeat_index": int(repeat_index),
        "data_patterns": variate["patterns"],
        "n_test_files": len(test_paths),
        "n_test_cosmologies": meta["n_test_cosmologies"],
        "n_dropped_nonfinite": n_bad_events,
        "test_ids_from_fixed_lock": meta["test_ids_from_fixed_lock"],
        "missing_params": missing_params,
        "excluded_params": exclude_params,
        "dropped_from_calibration": dropped,
        "num_posterior_samples": int(num_samples),
        "metrics": metrics,
    }
    tarp_intervals = _pop_credible_intervals(payload["metrics"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"misspec_evaluation_results_{cfg.match_string}.json"), "w") as f:
        json.dump(_to_json_compatible(payload), f, indent=4)
    if tarp_intervals:
        # Plot inputs: x = credible_intervals (alpha edges), y = ecp_bootstrap mean/std.
        intervals_payload = {
            "variate": name,
            "match_string": cfg.match_string,
            "available_params": [param_names[i] for i in available_idx],
            **tarp_intervals,
        }
        with open(os.path.join(out_dir, f"misspec_tarp_credible_intervals_{cfg.match_string}.json"), "w") as f:
            json.dump(_to_json_compatible(intervals_payload), f, indent=4)
    _save_posterior_samples(
        os.path.join(out_dir, f"misspec_posterior_samples_{cfg.match_string}.npz"),
        theta0s, samples, test_paths,
    )

    cal_full = metrics["tarp"]["full"]["calibration_error"]
    subset_key = "sigma_8__omega_m__w0"
    cal_subset = metrics["tarp"]["subsets"].get(subset_key, {}).get("calibration_error", float("nan"))
    print(f"[misspec] {name}: DONE cal_full={cal_full:.4f} cal_om_s8_w0={cal_subset:.4f} "
          f"fom={metrics.get('fom')} dMI={metrics.get('test_log_prob')}", flush=True)
    # Per-event posterior moments (scaled space, over the sample axis) for the cross-repeat
    # disagreement statistic; keyed by test-file basename for alignment across repeats.
    samp_np = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    per_event = {
        "mu": samp_np.mean(axis=0).astype(np.float32),
        "var": samp_np.var(axis=0).astype(np.float32),
        "test_files": [os.path.basename(p) for p in test_paths],
    }
    return {
        "n_test_files": len(test_paths),
        "n_dropped_nonfinite": n_bad_events,
        "available_params": [param_names[i] for i in available_idx],
        "cal_full": float(cal_full),
        "cal_om_s8_w0": float(cal_subset),
        "out_dir": out_dir,
        "_per_event": per_event,
    }
