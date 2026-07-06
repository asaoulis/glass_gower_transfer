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
)
from .evaluate_models import run_evaluation_on_samples
from ..data.data_selection import extract_cosmo_index


def _merged_preset(preset_overrides=None):
    """Global preset min/max boxes with optional per-run overrides (e.g. NLA-family a_ia)."""
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

    This is intended as a drop-in replacement for `build_gower_prior` when
    running inference that needs a non-uniform analytic prior.
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

def build_gower_prior(params, csv_path = "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv", preset_overrides=None):
    """Build the Gower Street prior in *scaled* space ([0,1]^D).

    - Empirical (flow) prior is trained/loaded for the subset of params not
      covered by the paper's analytic priors.
    - Known (analytic) priors from the paper are appended in the same order as
      they appear in `params`.
    """

    params = list(params)
    known_priors = build_gower_paper_known_priors()
    # Model-aware IA priors override the static (NLA-M) entries: NLA / NLA-z / TATT carry
    # different priors (and b_z / b_src), disambiguated by which companion param is present.
    known_priors = {**known_priors, **ia_marginal_priors(params)}

    # Preserve the caller's parameter ordering.
    extra_priors = {p: known_priors[p] for p in params if p in known_priors}
    flow_params = [p for p in params if p not in extra_priors]

    if len(flow_params) == 0:
        raise ValueError("build_gower_prior: no parameters left to model with a flow")

    preset = _merged_preset(preset_overrides)
    # Full scaler used to scale analytic priors into [0,1]
    full_scaler = _build_cosmo_preset_scaler(preset, params)
    # Flow scaler only needs the flow parameter subset
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


# ----------------------- Deterministic per-sim cosmology sampling -----------------------
# Used by the GLASS simulator to draw a fixed cosmology per sim_id so that many analysis
# variates (NLA-M, NLA-z, clustering, post-proc) share identical cosmologies — and so the
# expensive CAMB Cls can be cached per sim_id (see src/cosmology/mpi_camb.compute_or_load_glass_cls).

def _derive_cosmo_seed(base_seed, sim_num):
    """Stable 32-bit seed from (base_seed, sim_id) for reproducible per-sim cosmology draws."""
    ss = np.random.SeedSequence([int(base_seed), int(sim_num)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def make_seeded_cosmo_sampler(prior, column_names, scaler):
    """Wrap a scaled-space ([0,1]^D) torch/sbi prior into a deterministic cosmology sampler.

    Returns a callable ``sampler(sim_num, base_seed) -> dict[str, float]`` that:
      - seeds torch + numpy deterministically from (base_seed, sim_id),
      - draws ONE sample from ``prior`` (in scaled [0,1] space),
      - un-scales each event dim to physical units via ``scaler`` (min/max looked up by name),
        labelling the dims with ``column_names`` (the prior's event-dim order).

    ``column_names`` must match the prior's event-dim order: for ``build_gower_prior`` /
    ``build_flow_with_extras_prior`` this is ``flow_params + list(extra_priors)`` (no permutation);
    for ``build_analytic_prior`` it is the requested ``params`` order (it wraps a PermutedDistribution).
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

    Mirrors ``build_gower_prior``'s construction (analytic paper priors + empirical flow over the
    remaining params), then wraps it via ``make_seeded_cosmo_sampler`` so a given
    ``(sim_id, base_seed)`` always yields the same physical cosmology dict. ``params`` should be the
    cosmology parameter names only (e.g. omega_m, sigma_8, ombh2, h, ns, w0, mnu).
    """
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
    """Best-effort recovery of the ordered test-file list backing a loader, so posterior samples can be
    tagged with the sim/aug id of the test point each came from. Returns None if unavailable."""
    ds = getattr(test_loader, "dataset", None)
    for candidate in (ds, getattr(ds, "dataset", None)):
        paths = getattr(candidate, "paths", None)
        if paths is not None:
            return list(paths)
    return None


def _parse_aug_id(basename: str) -> int:
    """Trailing augmentation/noise index from an `output_<sim>_..._<aug>.h5` basename (-1 if absent)."""
    m = re.search(r"_(\d+)\.h5$", basename)
    return int(m.group(1)) if m else -1


def _save_posterior_samples(out_path, theta0s, samples, test_paths):
    """Persist raw posterior samples + true params next to the eval json, tagged with per-test-point
    sim/aug ids so runs can be checked for commensurability. theta0s ~ [N, D]; samples ~ [S, N, D];
    the id arrays align with axis-N. Positional path alignment assumes NO corrupt-file skips (true for
    the clean prebaked stores; H5CosmoDataset skips corrupt files, which would shift indices — guarded
    by a length check). Failures are non-fatal — the metrics json is the primary artifact."""
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

    Parameters
    ----------
    config : object
        Experiment configuration (must have base_path, experiment_name, etc.).
    test_loader : DataLoader
        Test dataloader to attach to the model.
    param_scaler : object
        Cosmology parameter scaler (with inverse_transform_minmax).
    reference_samples : np.ndarray or torch.Tensor, optional
        Reference posterior samples for comparison.
    model_builder : callable, optional
        Custom function to build a model given (config, test_loader).
        Defaults to src.ml.utils.build_model.
    """
    print("Running evaluation for experiment:", config.experiment_name, flush=True)
    our_prior = build_gower_prior(
        config.cosmo_param_names,
        preset_overrides=_config_preset_overrides(config),
    )
    sampling_kwargs["prior"] = our_prior
    # If ensemble evaluation is active, run a single ensemble evaluation for the
    # provided match_string (bound to repeat idx) and return early.
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

            # Save ensemble eval next to experiment folder (not per-run)
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

    # only keep folders that have config.match_string in their name
    if hasattr(config, "match_string") and config.match_string:
        run_folders = [f for f in run_folders if config.match_string in f]

    print(f"Running eval on {config.experiment_name} with {len(run_folders)} runs")

    # Default model builder if none provided
    if model_builder is None:
        def model_builder(cfg, loader):
            # Build a fresh model with the given test loader
            model = build_model(cfg, test_dataloader=loader)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            return model

    results = {}
    for run_folder in run_folders:
        print(run_folder, flush=True)

        # Find best checkpoint file and validation loss in this run folder
        best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
        if best_checkpoint is None:
            print(f"No valid checkpoint found in run folder: {run_folder}")
            continue

        # Temporarily set checkpoint_path on a shallow copy of the config to avoid
        # mutating the caller's config in-place across runs.
        from copy import copy
        cfg_for_run = copy(config)
        cfg_for_run.checkpoint_path = best_checkpoint

        # Build and prepare model using the provided or default model_builder
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

    Instead of loading a model per run folder and comparing their validation
    losses, we now:
      1. Scan all (matching) run folders for their best checkpoint and loss
         using `find_best_checkpoint`.
      2. Select the *global* best checkpoint across all these runs.
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

    # First pass: only look at checkpoint files / losses, do not instantiate models
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

    # Second pass: actually construct the model only for the globally best run
    model, best_model_path, _ = load_best_checkpoint_model(
        config, global_best_run_folder, data_parameters
    )

    if model is not None:
        print(
            f"Loaded best model from {best_model_path} with val loss {global_best_val_loss}"
        )
        scalers = None  # keep previous behaviour (scalers not used/returned yet)
        return model, scalers, best_model_path

    print("Failed to load model for the best checkpoint.")
    return None
