import os
import torch
import glob
import re
import json
from .tarp import get_tarp_coverage
from ..utils import load_best_checkpoint_model, build_model
from .loading_model import find_best_checkpoint, get_best_checkpoint

from .fom import compute_fom, compute_cov_matrix_per_sim

def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch=*-val_log_prob=*.ckpt"))
    
    best_checkpoint = None
    best_val_loss = float("inf")
    
    # Updated regex pattern to match any number of digits for epoch and allow negative float for loss
    loss_pattern = re.compile(r"checkpoint-epoch=(\d+)-val_log_prob=(-?\d+\.\d+).ckpt")
    
    for ckpt in checkpoint_files:
        match = loss_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            val_loss = float(match.group(2))  # Extract validation loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt
    
    return best_checkpoint, best_val_loss

def get_best_checkpoint(experiment_path, match_string):
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    print("Searching for best checkpoints in:", run_folders)
    best_checkpoints = []
    val_losses = []
    for run_folder in run_folders:
        if match_string not in run_folder:
            continue
        best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
        best_checkpoints.append(best_checkpoint)
        val_losses.append(best_val_loss)
    return best_checkpoints, val_losses


def compute_log_prob(model):
    with torch.no_grad():
        test_log_prob = model.compute_avg_log_prob()
    return {"test_log_prob": test_log_prob}
import torch

def rescale_parameters(tensor, scaler):
    """
    Rescales the given tensor using the provided scaler, ensuring the correct shape.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to be rescaled.
    scaler (object): The scaler object with an `inverse_transform_minmax` method.
    
    Returns:
    torch.Tensor: The rescaled tensor with the original shape.
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    original_shape = tensor.shape
    reshaped_tensor = tensor.reshape(-1, original_shape[-1])  # Flatten to (N, d) for scaling
    scaled_array = scaler.inverse_transform(reshaped_tensor.cpu().numpy())  # Apply inverse scaling
    scaled_tensor = torch.tensor(scaled_array, device=(device), dtype=torch.float32)  # Convert back to tensor
    return scaled_tensor.reshape(original_shape)  # Restore original shape

def generate_samples_and_run_eval(model, param_scaler, reference_samples=None, compute_calibration=True):
    
    theta0s, samples = model.generate_samples(model.test_dataloader, num_samples=10000)
    cosmological_params = param_scaler.parameter_names
    scaled_theta0s = rescale_parameters(theta0s, param_scaler)
    scaled_samples = rescale_parameters(samples, param_scaler)
    sample_means = scaled_samples.mean(axis=0)
    mse = torch.nn.functional.mse_loss(sample_means, scaled_theta0s, reduction='none')
    bias = scaled_samples.mean(axis=0) - scaled_theta0s
    # compute posterior ensemble per-parameter standard deviation
    std_devs = scaled_samples.std(axis=0)
    # also compute 68 and 95 CI widths per parameter
    width_68 = torch.quantile(scaled_samples, 0.84, dim=0) - torch.quantile(scaled_samples, 0.16, dim=0)
    width_95 = torch.quantile(scaled_samples, 0.975, dim=0) - torch.quantile(scaled_samples, 0.025, dim=0)

    eval_metrics = {
        "fom": compute_fom(samples),  # Compute Figure of Merit
        "sample_ensemble_mse": mse.mean().item(),  # Mean Squared Error of samples
    }
    per_dim_mse = mse.mean(dim=0).cpu().numpy()
    for dim, param_name in enumerate(cosmological_params):
        eval_metrics[param_name] = {}
        eval_metrics[param_name]["mse"] = per_dim_mse[dim].item()
        eval_metrics[param_name]["bias"] = bias.mean(dim=0)[dim].item()
        eval_metrics[param_name]["std_dev"] = std_devs.mean(dim=0)[dim].item()
        eval_metrics[param_name]["width_68"] = width_68.mean(dim=0)[dim].item()
        eval_metrics[param_name]["width_95"] = width_95.mean(dim=0)[dim].item()
    if "omega_m" in cosmological_params and "sigma_8" in cosmological_params:
        i_sigma8 = cosmological_params.index("sigma_8")
        i_omegam = cosmological_params.index("omega_m")
        s8_samples = (
            scaled_samples[:, :, i_sigma8]
            * (scaled_samples[:, :, i_omegam] / 0.3) ** 0.5
        )  # (N_cosmo, N_samples)
        s8_theta0s = (
            scaled_theta0s[:, i_sigma8]
            * (scaled_theta0s[:, i_omegam] / 0.3) ** 0.5
        )  # (N_cosmo,)

        s8_mean = s8_samples.mean(dim=0)  # mean over samples → (N_cosmo,)

        eval_metrics["s8"] = {}
        eval_metrics["s8"]["mse"] = torch.mean((s8_mean - s8_theta0s) ** 2).item()
        eval_metrics["s8"]["bias"] = (s8_mean - s8_theta0s).mean().item()
        eval_metrics["s8"]["std_dev"] = s8_samples.std(dim=0).mean().item()
        eval_metrics["s8"]["width_68"] = (
            torch.quantile(s8_samples, 0.84, dim=0)
            - torch.quantile(s8_samples, 0.16, dim=0)
        ).mean().item()
        eval_metrics["s8"]["width_95"] = (
            torch.quantile(s8_samples, 0.975, dim=0)
            - torch.quantile(s8_samples, 0.025, dim=0)
        ).mean().item()
    cov_matrices = compute_cov_matrix_per_sim(scaled_samples)
    inv_covariances = torch.linalg.inv(cov_matrices)

    mahalanobis_distances = torch.sqrt(torch.einsum('bi,bij,bj->b', bias, inv_covariances, bias))
    eval_metrics['mahalanobis_distance_mean'] = mahalanobis_distances.mean().item()
    eval_metrics['mahalanobis_distance_std'] = mahalanobis_distances.std().item()

    if compute_calibration:
        coverage = get_tarp_coverage(samples.cpu().numpy(), theta0s.cpu().numpy(), bootstrap=True, num_bootstrap=25)
        rank_histogram = np.diff(coverage[0].mean(axis=0))
        rank_histogram *= len(rank_histogram)
        expected_ranks = np.ones(len(rank_histogram))
        calibration_error = np.mean((rank_histogram - expected_ranks)**2) 
        eval_metrics.update({
            "calibration_error": calibration_error
        })
    if reference_samples is not None:
        reference_samples = reference_samples[:, :samples.shape[1]]
        scaled_reference_samples = rescale_parameters(torch.tensor(reference_samples), param_scaler)
        eval_metrics.update({
            "ref_post_mean_mse": torch.nn.functional.mse_loss(
                scaled_samples.mean(axis=0), scaled_reference_samples.mean(axis=0)
            ).item(),
            "ref_post_cov_mse": torch.nn.functional.mse_loss(
                compute_cov_matrix_per_sim(scaled_samples), compute_cov_matrix_per_sim(scaled_reference_samples)
            ).item()
        })

    return eval_metrics


def evaluate_best_checkpoint(config, test_loader, param_scaler, reference_samples=None, model_builder=None):
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

    experiment_path = f"{config.base_path}/checkpoints/{config.experiment_name}/{config.experiment_name}"
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
        eval_metrics = generate_samples_and_run_eval(model, param_scaler, reference_samples)

        results[run_folder] = {
            "best_checkpoint": best_checkpoint,
            "best_val_loss": best_val_loss,
            "metrics": {**metrics, **eval_metrics},
        }

        results_path = os.path.join(run_folder, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results[run_folder], f, indent=4)

    return results

import os
import json
import numpy as np
from pathlib import Path

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(d, sep="."):
    result = {}
    for k, v in d.items():
        keys = k.split(sep)
        cur = result
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = v
    return result

import os
import json
import numpy as np
from pathlib import Path


def parse_results(experiment_name, base_path="/share/gpu0/asaoulis/cmd/checkpoints"):
    experiment_path = os.path.join(base_path, f"{experiment_name}/{experiment_name}")
    # check if experiment path exists
    run_folders = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    results = {}
    aggregated_results = {}
    ensemble_results = {}

    # ---- Parse ensemble results (no aggregation) ----
    ensemble_result_paths = Path(experiment_path).glob("ensemble_evaluation_results_*.json")
    for ensemble_result in ensemble_result_paths:
        match_string = ensemble_result.name.split("_")[-1].split(".")[0]
        with open(ensemble_result, "r") as f:
            data = json.load(f)
            ensemble_results[f"ensemble_{match_string}"] = data["metrics"]

    # ---- Parse individual runs ----
    for run_folder in run_folders:
        results_file = os.path.join(run_folder, "evaluation_results.json")
        if not os.path.exists(results_file):
            continue

        with open(results_file, "r") as f:
            data = json.load(f)

        run_name = os.path.basename(run_folder)
        metrics = flatten_dict(data["metrics"])
        results[run_name] = metrics

    # ---- Aggregate by base run name ----
    for run_name, metrics in results.items():
        base_name = (
            "_".join(run_name.split("_")[:-1])
            if run_name.split("_")[-1].isdigit()
            else run_name
        )

        aggregated_results.setdefault(base_name, {})

        for metric, value in metrics.items():
            aggregated_results[base_name].setdefault(metric, [])
            aggregated_results[base_name][metric].append(value)

    # ---- Compute mean + stderr ----
    final_results = {}

    # Ensemble results: copy as-is
    for ensemble_name, metrics in ensemble_results.items():
        final_results[ensemble_name] = metrics

    for base_name, metrics in aggregated_results.items():
        final_results[base_name] = {}

        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                mean, stderr = np.nan, np.nan
            elif len(values) == 1:
                mean, stderr = values[0], 0.0
            else:
                mean = np.mean(values)
                stderr = np.std(values, ddof=1) / np.sqrt(len(values))

            final_results[base_name][metric] = {
                "mean": mean,
                "stderr": stderr,
            }

    return final_results
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
        return model, scalers

    print("Failed to load model for the best checkpoint.")
    return None
