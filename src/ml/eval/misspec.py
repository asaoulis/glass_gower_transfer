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
    {"name": "nla_m", "patterns": f"{_GPU5}/gower_mocks_nla_m_f16_fwhm4_lmin56_lcut1400/output_*.h5",
     "exclude_params": [], "in_distribution": True},
    {"name": "nla", "patterns": f"{_GPU4}/gower_mocks_nla/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "nla_z", "patterns": f"{_GPU4}/gower_mocks_nla_z/output_*.h5", "exclude_params": ["a_ia"]},
    {"name": "gb0p5", "patterns": f"{_GPU4}/gower_mocks_gb0p5/output_*.h5", "exclude_params": []},
    {"name": "gb1p5", "patterns": f"{_GPU4}/gower_mocks_gb1p5/output_*.h5", "exclude_params": []},
]


def _load_experiment_config(experiment_name: str):
    """Rebuild the experiment config exactly like eval.py's load_config + list-branch handling."""
    from config.default import get_default_config
    from config.experiments import experiments as base_experiments
    from config.ablations import ablation_experiments
    from config.kids_legacy import kids_legacy_experiments

    exps = dict(base_experiments)
    exps.update(ablation_experiments)
    exps.update(kids_legacy_experiments)
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
    fixed_test_sim_ids,
    test_shape_noise_idx=(0, (0, 1)),
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Variate TEST loader with the ORIGINAL scalers injected (never refit).

    Test cosmologies = (fixed-test lock ids ∩ on-disk ids); falls back to ALL on-disk
    cosmologies when there is no overlap (e.g. a small gb subset outside the lock range).
    """
    all_paths = collect_paths(patterns)
    by_cosmo: Dict[int, List[str]] = {}
    for p in all_paths:
        by_cosmo.setdefault(extract_cosmo_index(p), []).append(p)
    for c in by_cosmo:
        by_cosmo[c].sort()

    lock_ids = resolve_fixed_test_ids(fixed_test_sim_ids) or set()
    test_ids = sorted(set(by_cosmo.keys()) & lock_ids)
    used_fallback = False
    if not test_ids:
        test_ids = sorted(by_cosmo.keys())
        used_fallback = True
        print(f"[misspec] WARNING: no overlap with the {len(lock_ids)} fixed test ids; "
              f"falling back to ALL {len(test_ids)} on-disk cosmologies (model may have seen "
              "these cosmologies at nla_m physics).", flush=True)

    test_paths = [p for c in test_ids for p in by_cosmo[c]]
    filtered = _filter_paths_by_shape_noise_idx(test_paths, list(test_shape_noise_idx))
    if not filtered:
        print(f"[misspec] WARNING: shape-noise filter {test_shape_noise_idx} matched no files; "
              "using all test-cosmology files.", flush=True)
        filtered = test_paths

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
    repeat_index: int = 0,
    variates: Optional[List[Dict]] = None,
    num_samples: int = 10000,
    prior_num_samples: int = 20_000,
    test_shape_noise_idx=(0, (0, 1)),
    out_subdir: str = "misspec",
):
    from ..models.utils import apply_repeat_config

    variates = DEFAULT_VARIATES if variates is None else variates

    # --- base config bound to the requested repeat ----------------------------------------
    cfg = _load_experiment_config(base_experiment)
    # Same test-point sub-selection for the in-distribution reference as for every variate
    # (rot0, inner noise {0,1}) so coverage curves are directly comparable.
    cfg.test_shape_noise_idx = list(test_shape_noise_idx)
    repeat_match, _ = apply_repeat_config(cfg, repeat_index)
    cfg.match_string = repeat_match
    param_names = list(cfg.cosmo_param_names)
    eb_variant = getattr(cfg, "eb_map_variant", None)
    nested_keys = build_nested_keys_from_quantities(list(cfg.dataset_quantities), eb_variant)
    print(f"[misspec] base experiment '{base_experiment}' repeat={repeat_index} "
          f"match_string={cfg.match_string} params={param_names} eb_variant={eb_variant}",
          flush=True)

    # --- ORIGINAL scalers: fit once on the nla_m train+val split (ensemble path) ----------
    scalers, _, _, in_dist_test_loader = prepare_data_parameters(cfg)
    orig_key_scalers = scalers["data"]
    orig_cosmo_scaler = scalers["cosmo"]
    print(f"[misspec] original scalers rebuilt from {cfg.data_patterns} "
          f"(data keys: {sorted(orig_key_scalers)})", flush=True)

    # --- r0 ensemble model, built ONCE (loaders are swapped per variate) ------------------
    n_ens = int(getattr(cfg, "ensemble_repeats", 1) or 1)
    model = build_ensemble_model_from_checkpoints(
        cfg,
        in_dist_test_loader,
        match_string=cfg.match_string,
        member_test_loaders=[in_dist_test_loader] * n_ens,
    )
    if model is None:
        raise RuntimeError(f"Could not build the '{base_experiment}' {cfg.match_string} ensemble "
                           "(no checkpoints found?)")

    # --- base Gower prior + one shared set of prior samples (FoM shrinkage reference) -----
    prior = build_gower_prior(param_names, preset_overrides=_config_preset_overrides(cfg))
    prior_samples_scaled = _sample_from_prior(prior, prior_num_samples, target_dim=len(param_names))

    out_root = os.path.join(cfg.base_path, "checkpoints", cfg.experiment_name, out_subdir)
    summary = {}
    for variate in variates:
        name = variate["name"]
        try:
            result = _eval_one_variate(
                variate, model, cfg, nested_keys, param_names,
                orig_key_scalers, orig_cosmo_scaler, prior_samples_scaled,
                num_samples=num_samples,
                test_shape_noise_idx=test_shape_noise_idx,
                out_dir=os.path.join(out_root, name),
            )
            summary[name] = result
        except Exception as e:
            print(f"[misspec] {name}: FAILED — {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            summary[name] = {"error": f"{type(e).__name__}: {e}"}

    print("\n[misspec] ============ SUMMARY ============", flush=True)
    for name, res in summary.items():
        if "error" in res:
            print(f"[misspec] {name}: ERROR {res['error']}", flush=True)
        else:
            print(f"[misspec] {name}: n_test={res['n_test_files']} "
                  f"cal_full={res['cal_full']:.4f} cal_om_s8_w0={res['cal_om_s8_w0']:.4f} "
                  f"available={res['available_params']}", flush=True)
    return summary


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
        fixed_test_sim_ids=getattr(cfg, "fixed_test_sim_ids", None),
        test_shape_noise_idx=test_shape_noise_idx,
        # Cap at 64: train batch sizes (100-128) are l40s-sized; eval may land on a smaller GPU.
        batch_size=min(64, int(getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", 64))),
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

    # Swap the test set under the prebuilt ensemble: the ensemble-level loader feeds theta0s
    # and compute_avg_log_prob; each member's loader feeds its own encode+sample pass.
    model.test_dataloader = loader
    for m in model.members:
        m.test_dataloader = loader

    theta0s, samples = model.generate_samples(num_samples=num_samples)

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
        "data_patterns": variate["patterns"],
        "n_test_files": meta["n_test_files"],
        "n_test_cosmologies": meta["n_test_cosmologies"],
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
        theta0s, samples, meta["test_paths"],
    )

    cal_full = metrics["tarp"]["full"]["calibration_error"]
    subset_key = "sigma_8__omega_m__w0"
    cal_subset = metrics["tarp"]["subsets"].get(subset_key, {}).get("calibration_error", float("nan"))
    print(f"[misspec] {name}: DONE cal_full={cal_full:.4f} cal_om_s8_w0={cal_subset:.4f} "
          f"fom={metrics.get('fom')} dMI={metrics.get('test_log_prob')}", flush=True)
    return {
        "n_test_files": meta["n_test_files"],
        "available_params": [param_names[i] for i in available_idx],
        "cal_full": float(cal_full),
        "cal_om_s8_w0": float(cal_subset),
        "out_dir": out_dir,
    }
