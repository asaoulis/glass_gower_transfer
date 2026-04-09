import os
import torch
import glob
import re
import json
import numpy as np

from torch.distributions import Uniform

from src.ml.utils import _build_cosmo_preset_scaler
from src.ml.data.constants import COSMO_PARAM_PRESET_MINMAX
from ..utils import load_best_checkpoint_model, build_model
from .loading_model import find_best_checkpoint
from ..utils import is_ensemble_eval_active, build_ensemble_model_from_checkpoints
from ..data.priors import train_or_load_gower_prior,build_flow_with_extras_prior
from .evaluate_models import run_evaluation_on_samples


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

def build_gower_prior(params, csv_path = "/home/asaoulis/projects/glass_transfer/kids-legacy-sbi/data/gower_st/PKDGRAV3_on_DiRAC_DES_330.csv"):
    extra_priors = {
        "a_ia": Uniform(
            torch.tensor([4.48]),
            torch.tensor([7.0]),
        ),
        "b_ia": Uniform(
            torch.tensor([0.28]),
            torch.tensor([0.6]),
        ),
    }

    scaler = _build_cosmo_preset_scaler(COSMO_PARAM_PRESET_MINMAX, params)
    flow = train_or_load_gower_prior(
        csv_path=csv_path,
        variables=params,
        scaler=scaler,
        drop_first=192,
    )
    # remove extra_priors from params to get flow params
    flow_params = [ p for p in params if p not in extra_priors ]
    prior = build_flow_with_extras_prior(flow, flow_params, scaler, extra_priors=extra_priors)
    return prior

def generate_samples_and_run_eval(model, param_scaler, reference_samples=None, compute_calibration=True, **sampling_kwargs):
    num_samples = sampling_kwargs.pop("num_samples", 10000)
    theta0s, samples = model.generate_samples(num_samples=num_samples, **sampling_kwargs)
    eval_metrics = run_evaluation_on_samples(theta0s, samples, param_scaler, reference_samples=reference_samples, compute_calibration=compute_calibration)
    return eval_metrics


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

    # If ensemble evaluation is active, run a single ensemble evaluation for the
    # provided match_string (bound to repeat idx) and return early.
    if is_ensemble_eval_active(config) and getattr(config, "match_string", None):
        match_string = config.match_string
        ensemble_model = build_ensemble_model_from_checkpoints(
            config,
            test_loader,
            match_string=match_string,
            member_test_loaders=ensemble_member_test_loaders,
        )
        if ensemble_model is None:
            print("Failed to build ensemble model; falling back to single-run evaluation.")
        else:
            metrics = compute_log_prob(ensemble_model)
            eval_metrics = generate_samples_and_run_eval(
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
    is_nle_inference_mode = config.inference_mode == "nle"
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
        if is_nle_inference_mode:
            our_prior = build_gower_prior(cfg_for_run.cosmo_param_names)
            sampling_kwargs["prior"] = our_prior
        eval_metrics = generate_samples_and_run_eval(model, param_scaler, reference_samples, **sampling_kwargs)

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
