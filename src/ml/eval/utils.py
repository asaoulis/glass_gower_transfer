"""Evaluation helpers.

Gower and S8-analytic priors in scaled space, log-prob and credible-interval computation,
best-checkpoint loading, and the posterior-sample dumps written next to each eval json.
"""
import os
import torch
import glob
import re
import json
import numpy as np

from src.ml.utils import _build_cosmo_preset_scaler
from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
from ..utils import load_best_checkpoint_model, build_model
from .loading_model import find_best_checkpoint
from ..utils import is_ensemble_eval_active, build_ensemble_model_from_checkpoints
from ..data.priors import (
    train_or_load_gower_prior,
    build_flow_with_extras_prior,
    build_gower_paper_known_priors,
    build_analytic_prior,
    ia_marginal_priors,
    galaxy_bias_marginal_priors,
    _infer_galaxy_bias_kappa,
)
from .evaluate_models import run_evaluation_on_samples
from ..data.data_selection import extract_cosmo_index


def _merged_preset(preset_overrides=None):
    """Global preset min/max boxes with optional per-run overrides, such as the NLA-family a_ia."""
    preset = dict(COSMO_PARAM_PRESET_MINMAX)
    if preset_overrides:
        preset.update({k: tuple(v) for k, v in preset_overrides.items()})
    return preset


def _config_preset_overrides(config):
    """Extract scaler_options['cosmo']['preset_overrides'] from a config, if present."""
    scaler_options = getattr(config, "scaler_options", None) or {}
    cosmo_opts = scaler_options.get("cosmo", {}) if hasattr(scaler_options, "get") else {}
    return cosmo_opts.get("preset_overrides") if hasattr(cosmo_opts, "get") else None


def build_s8_analytic_prior(params, scaler=None, *, return_restricted=None, preset_overrides=None):
    """Build the S8-box analytic prior in scaled space, in `params` order.

    A drop-in replacement for build_gower_prior.
    """
    params = list(params)
    if scaler is None:
        scaler = _build_cosmo_preset_scaler(_merged_preset(preset_overrides), params)
    if scaler is None:
        raise ValueError("build_s8_analytic_prior: scaler is None")

    if return_restricted is None:
        # Only needed when the prior has unbounded scaled support (Gaussian blocks): NLA-M
        # (a_ia, b_ia) and NLA-z's b_z. Bounded IA priors (NLA a_ia, TATT b_src) don't need it.
        return_restricted = ("b_ia" in params) or ("b_z" in params)

    return build_analytic_prior(
        params,
        scaler,
        return_restricted=bool(return_restricted),
    )


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pop_credible_intervals(node):
    if not isinstance(node, dict):
        return None

    interval_like_keys = {"credible_intervals", "ecp_bootstrap", "ecp"}
    extracted = {}
    for key in list(node.keys()):
        value = node[key]
        if key in interval_like_keys:
            extracted[key] = node.pop(key)
            continue

        child = _pop_credible_intervals(value)
        if child:
            extracted[key] = child

    return extracted

def compute_log_prob(model):
    with torch.no_grad():
        test_log_prob = model.compute_avg_log_prob()
    return {"test_log_prob": test_log_prob}

def _resolve_prior_csv(csv_path):
    """Resolve the Gower prior CSV: csv_path if it exists, else $GLASS_GOWER_PRIOR_CSV, else the repo copy."""
    if csv_path and os.path.exists(csv_path):
        return csv_path
    env = os.environ.get("GLASS_GOWER_PRIOR_CSV")
    if env and os.path.exists(env):
        return env
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    local = os.path.join(repo, "kids-legacy-sbi", "data", "gower_st", "PKDGRAV3_on_DiRAC_DES_330.csv")
    return local if os.path.exists(local) else csv_path


def build_gower_prior(params, csv_path = "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv", preset_overrides=None):
    """Build the Gower Street prior in scaled space ([0,1]^D).

    An empirical flow covers the params without an analytic paper prior. The analytic priors
    follow in `params` order.
    """

    csv_path = _resolve_prior_csv(csv_path)
    params = list(params)
    known_priors = build_gower_paper_known_priors()
    # Model-aware IA priors override the static NLA-M entries, chosen by the companion param.
    known_priors = {**known_priors, **ia_marginal_priors(params)}
    # b_g_bin* are analytic truncated Gaussians whose width kappa is recovered from the caller's
    # preset_overrides, so the prior matches the model's boxes.
    _bg_kappa = _infer_galaxy_bias_kappa(preset_overrides)
    known_priors = {**known_priors, **galaxy_bias_marginal_priors(params, kappa=_bg_kappa)}

    # Preserve the caller's parameter ordering.
    extra_priors = {p: known_priors[p] for p in params if p in known_priors}
    flow_params = [p for p in params if p not in extra_priors]

    preset = _merged_preset(preset_overrides)
    full_scaler = _build_cosmo_preset_scaler(preset, params)

    # With every parameter analytic there is no flow: the prior is the product of the marginals.
    flow = None
    if flow_params:
        flow_scaler = _build_cosmo_preset_scaler(preset, flow_params)
        flow = train_or_load_gower_prior(
            csv_path=csv_path,
            variables=flow_params,
            scaler=flow_scaler,
            drop_first=192,
        )

    prior = build_flow_with_extras_prior(
        flow,
        flow_params,
        full_scaler,
        extra_priors=extra_priors,
    )
    return prior


# --- deterministic per-sim cosmology sampling ---

def _derive_cosmo_seed(base_seed, sim_num):
    """Stable 32-bit seed from (base_seed, sim_id) for reproducible per-sim cosmology draws."""
    ss = np.random.SeedSequence([int(base_seed), int(sim_num)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def make_seeded_cosmo_sampler(prior, column_names, scaler):
    """Wrap a scaled-space ([0,1]^D) torch/sbi prior into a deterministic cosmology sampler.

    Returns sampler(sim_num, base_seed) -> dict[str, float]: one draw seeded from
    (base_seed, sim_id), un-scaled to physical units by name. column_names must follow the prior's
    event-dim order: flow_params + list(extra_priors) for build_gower_prior, the requested params
    for build_analytic_prior.
    """
    column_names = list(column_names)
    name_to_idx = {n: i for i, n in enumerate(scaler.parameter_names)}
    mins = np.asarray(scaler.min, dtype=np.float64)
    maxs = np.asarray(scaler.max, dtype=np.float64)

    def sampler(sim_num, base_seed):
        seed = _derive_cosmo_seed(base_seed, sim_num)
        torch.manual_seed(seed)
        np.random.seed(seed)
        sample = prior.sample((1,))
        arr = np.asarray(sample.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if arr.shape[0] != len(column_names):
            raise ValueError(
                f"make_seeded_cosmo_sampler: prior returned {arr.shape[0]} event dims but "
                f"{len(column_names)} column names were given ({column_names})."
            )
        out = {}
        for j, name in enumerate(column_names):
            idx = name_to_idx[name]
            out[name] = float(arr[j] * (maxs[idx] - mins[idx]) + mins[idx])
        return out

    return sampler


def build_cosmo_param_sampler(
    params,
    csv_path="/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv",
    preset_overrides=None,
):
    """Deterministic per-sim cosmology sampler backed by the Gower Street flow prior.

    Built as build_gower_prior and wrapped by make_seeded_cosmo_sampler. params holds the
    cosmology parameter names only.
    """
    csv_path = _resolve_prior_csv(csv_path)
    params = list(params)
    known_priors = build_gower_paper_known_priors()
    known_priors = {**known_priors, **ia_marginal_priors(params)}
    extra_priors = {p: known_priors[p] for p in params if p in known_priors}
    flow_params = [p for p in params if p not in extra_priors]
    if len(flow_params) == 0:
        raise ValueError("build_cosmo_param_sampler: no parameters left to model with a flow")

    preset = _merged_preset(preset_overrides)
    full_scaler = _build_cosmo_preset_scaler(preset, params)
    flow_scaler = _build_cosmo_preset_scaler(preset, flow_params)
    flow = train_or_load_gower_prior(
        csv_path=csv_path,
        variables=flow_params,
        scaler=flow_scaler,
        drop_first=192,
    )
    prior = build_flow_with_extras_prior(
        flow,
        flow_params,
        full_scaler,
        extra_priors=extra_priors,
    )
    # build_flow_with_extras_prior emits event dims as [flow columns..., extra keys...].
    column_names = list(flow_params) + list(extra_priors.keys())
    return make_seeded_cosmo_sampler(prior, column_names, full_scaler)


def _resolve_test_paths(test_loader):
    """Ordered test-file list backing a loader, through .dataset and .base_ds wrappers, or None."""
    def _walk(ds, depth=0):
        if ds is None or depth > 3:
            return None
        paths = getattr(ds, "paths", None)
        if paths is not None:
            return list(paths)
        for attr in ("dataset", "base_ds"):
            found = _walk(getattr(ds, attr, None), depth + 1)
            if found is not None:
                return found
        return None

    return _walk(getattr(test_loader, "dataset", None))


def _parse_aug_id(basename: str) -> int:
    """Trailing augmentation/noise index from an `output_<sim>_..._<aug>.h5` basename (-1 if absent)."""
    m = re.search(r"_(\d+)\.h5$", basename)
    return int(m.group(1)) if m else -1


def _save_posterior_samples(out_path, theta0s, samples, test_paths):
    """Save raw posterior samples and true params next to the eval json, with per-test-point sim/aug ids.

    theta0s [N, D], samples [S, N, D], the id arrays align with axis N. Path alignment is
    positional, guarded by a length check. Failures only warn."""
    try:
        theta_np = theta0s.detach().cpu().numpy() if hasattr(theta0s, "detach") else np.asarray(theta0s)
        samp_np = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        payload = {"samples": samp_np, "theta0s": theta_np}
        n = theta_np.shape[0]
        if test_paths is not None and len(test_paths) == n:
            files = [os.path.basename(p) for p in test_paths]
            payload["test_files"] = np.array(files)
            payload["sim_ids"] = np.array([extract_cosmo_index(p) for p in test_paths], dtype=np.int64)
            payload["aug_ids"] = np.array([_parse_aug_id(f) for f in files], dtype=np.int64)
        elif test_paths is not None:
            print(f"[save-samples] path/theta count mismatch ({len(test_paths)} vs {n}); saving samples "
                  "WITHOUT sim/aug ids (corrupt-file skip or wrong loader?).", flush=True)
        else:
            print("[save-samples] test paths unavailable; saving samples without sim/aug ids.", flush=True)
        np.savez_compressed(out_path, **payload)
        print(f"[save-samples] wrote {out_path}  (N={n}, keys={sorted(payload)})", flush=True)
    except Exception as e:  # never let a sample dump break the eval
        print(f"[save-samples] WARNING: failed to save posterior samples to {out_path}: {e}", flush=True)


def generate_samples_and_run_eval(model, param_scaler, reference_samples=None, compute_calibration=True, **sampling_kwargs):
    num_samples = sampling_kwargs.pop("num_samples", 10000)
    prior = sampling_kwargs.get("prior", None)
    prior_num_samples = sampling_kwargs.pop("prior_num_samples", 20_000)
    theta0s, samples = model.generate_samples(num_samples=num_samples, **sampling_kwargs)
    eval_metrics = run_evaluation_on_samples(
        theta0s,
        samples,
        param_scaler,
        reference_samples=reference_samples,
        compute_calibration=compute_calibration,
        prior=prior,
        prior_num_samples=prior_num_samples,
    )
    # Also return the raw draws + true params so callers can persist them with sim/aug ids.
    return eval_metrics, theta0s, samples


def evaluate_best_checkpoint(
    config,
    test_loader,
    param_scaler,
    reference_samples=None,
    model_builder=None,
    ensemble_member_test_loaders=None,
    **sampling_kwargs,
):
    """Evaluate all matching run folders for an experiment.

    test_loader is attached to each model. param_scaler is the cosmology scaler.
    reference_samples are optional reference posterior samples. model_builder(config, test_loader)
    defaults to src.ml.utils.build_model.
    """
    print("Running evaluation for experiment:", config.experiment_name, flush=True)
    our_prior = build_gower_prior(
        config.cosmo_param_names,
        preset_overrides=_config_preset_overrides(config),
    )
    sampling_kwargs["prior"] = our_prior
    # Optional MCMC pool size (config.sampling_num_jobs). A caller's num_jobs still wins.
    _sampling_num_jobs = getattr(config, "sampling_num_jobs", None)
    if _sampling_num_jobs is not None and "num_jobs" not in sampling_kwargs:
        sampling_kwargs["num_jobs"] = int(_sampling_num_jobs)
    # Ensemble evaluation: one evaluation for config.match_string, then return.
    if is_ensemble_eval_active(config) and getattr(config, "match_string", None):
        match_string = config.match_string
        ensemble_model = build_ensemble_model_from_checkpoints(
            config,
            test_loader,
            match_string=match_string,
            member_test_loaders=ensemble_member_test_loaders,
            model_builder=model_builder,
        )
        if ensemble_model is None:
            print("Failed to build ensemble model; falling back to single-run evaluation.")
        else:
            metrics = compute_log_prob(ensemble_model)
            eval_metrics, theta0s, samples = generate_samples_and_run_eval(
                ensemble_model,
                param_scaler,
                reference_samples,
                **sampling_kwargs,
            )
            metrics_payload = {**metrics, **eval_metrics}
            tarp_intervals = _pop_credible_intervals(metrics_payload)

            # The ensemble json goes in the experiment folder, not a run folder.
            experiment_path = f"{config.base_path}/checkpoints/{config.experiment_name}/"
            os.makedirs(experiment_path, exist_ok=True)
            results_path = os.path.join(
                experiment_path,
                f"ensemble_evaluation_results_{match_string}.json",
            )
            out = {
                "match_string": match_string,
                "ensemble_repeats": int(getattr(config, "ensemble_repeats", 1) or 1),
                "metrics": metrics_payload,
            }
            with open(results_path, "w") as f:
                json.dump(_to_json_compatible(out), f, indent=4)

            if tarp_intervals:
                intervals_path = os.path.join(
                    experiment_path,
                    f"ensemble_tarp_credible_intervals_{match_string}.json",
                )
                with open(intervals_path, "w") as f:
                    json.dump(_to_json_compatible(tarp_intervals), f, indent=4)

            _save_posterior_samples(
                os.path.join(experiment_path, f"ensemble_posterior_samples_{match_string}.npz"),
                theta0s, samples, _resolve_test_paths(test_loader),
            )
            return {"ensemble": out}

    experiment_path = f"{config.base_path}/checkpoints/{config.experiment_name}/"
    if not os.path.exists(experiment_path):
        print("Experiment path does not exist:", experiment_path, flush=True)
        return

    run_folders = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    if hasattr(config, "match_string") and config.match_string:
        run_folders = [f for f in run_folders if config.match_string in f]

    print(f"Running eval on {config.experiment_name} with {len(run_folders)} runs")

    if model_builder is None:
        def model_builder(cfg, loader):
            model = build_model(cfg, test_dataloader=loader)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            return model

    results = {}
    for run_folder in run_folders:
        print(run_folder, flush=True)

        best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
        if best_checkpoint is None:
            print(f"No valid checkpoint found in run folder: {run_folder}")
            continue

        # A shallow copy, so the caller's config keeps its checkpoint_path across runs.
        from copy import copy
        cfg_for_run = copy(config)
        cfg_for_run.checkpoint_path = best_checkpoint

        model = model_builder(cfg_for_run, test_loader)

        if model is None:
            print(f"Failed to build model for run folder: {run_folder}")
            continue

        metrics = compute_log_prob(model)
        eval_metrics, theta0s, samples = generate_samples_and_run_eval(model, param_scaler, reference_samples, **sampling_kwargs)

        metrics_payload = {**metrics, **eval_metrics}
        tarp_intervals = _pop_credible_intervals(metrics_payload)

        results[run_folder] = {
            "best_checkpoint": best_checkpoint,
            "best_val_loss": best_val_loss,
            "metrics": metrics_payload,
        }

        results_path = os.path.join(run_folder, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(_to_json_compatible(results[run_folder]), f, indent=4)

        if tarp_intervals:
            intervals_path = os.path.join(run_folder, "tarp_credible_intervals.json")
            with open(intervals_path, "w") as f:
                json.dump(_to_json_compatible(tarp_intervals), f, indent=4)

        _save_posterior_samples(
            os.path.join(run_folder, "posterior_samples.npz"),
            theta0s, samples, _resolve_test_paths(test_loader),
        )

    return results

def load_best_model_and_build_posterior(config, ds_string_match="", data_parameters=None):
    """Load the single best model across all matching run folders.

    Only the globally best checkpoint (by find_best_checkpoint) is instantiated.
    """
    patterns = [ f"{config.base_path}/checkpoints/{config.experiment_name}/run_{config.experiment_name}",
                f"{config.base_path}/checkpoints/{config.experiment_name}/{config.experiment_name}",
                 f"{config.base_path}/checkpoints/{config.experiment_name}" ]
    run_folders = []
    for experiment_path in patterns:
        if not os.path.exists(experiment_path):
            continue
        run_folders += [
            os.path.join(experiment_path, d)
            for d in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, d))
        ]
    print(
        f"Loading best model for {config.experiment_name} from {experiment_path} with {len(run_folders)} runs"
    )

    global_best_val_loss = float("inf")
    global_best_run_folder = None

    for run_folder in run_folders:
        if ds_string_match and ds_string_match not in run_folder:
            continue

        best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
        if best_checkpoint is None:
            continue

        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            global_best_run_folder = run_folder

    if global_best_run_folder is None:
        print("No valid checkpoints found.")
        return None

    model, best_model_path, _ = load_best_checkpoint_model(
        config, global_best_run_folder, data_parameters
    )

    if model is not None:
        print(
            f"Loaded best model from {best_model_path} with val loss {global_best_val_loss}"
        )
        scalers = None
        return model, scalers, best_model_path

    print("Failed to load model for the best checkpoint.")
    return None
