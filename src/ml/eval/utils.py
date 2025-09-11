import os
import torch
import glob
import re
import json
from ..utils import prepare_data_and_model
from .tarp import get_tarp_coverage

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
    original_shape = tensor.shape
    reshaped_tensor = tensor.reshape(-1, original_shape[-1])  # Flatten to (N, d) for scaling
    scaled_array = scaler.inverse_transform_minmax(reshaped_tensor.cpu().numpy())  # Apply inverse scaling
    scaled_tensor = torch.tensor(scaled_array, device='cuda', dtype=torch.float32)  # Convert back to tensor
    return scaled_tensor.reshape(original_shape)  # Restore original shape

def generate_samples_and_run_eval(model, param_scaler, reference_samples=None, compute_calibration=True):
    
    theta0s, samples = model.generate_samples(model.test_dataloader, num_samples=10000)
    
    scaled_theta0s = rescale_parameters(theta0s, param_scaler)
    scaled_samples = rescale_parameters(samples, param_scaler)
    sample_means = scaled_samples.mean(axis=0)
    mse = torch.nn.functional.mse_loss(sample_means, scaled_theta0s, reduction='none')
    bias = scaled_samples.mean(axis=0) - scaled_theta0s

    eval_metrics = {
        "fom": compute_fom(samples),  # Compute Figure of Merit
        "sample_ensemble_mse": mse.mean().item(),  # Mean Squared Error of samples
    }
    per_dim_mse = mse.mean(dim=0).cpu().numpy()
    for dim in range(theta0s.shape[1]):
        eval_metrics[f"mse_{dim}"] = per_dim_mse[dim].item()
        eval_metrics[f"bias_{dim}"] = bias.mean(dim=0)[dim].item()
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


def load_best_checkpoint_model(config, run_folder, data_parameters=None):
    """Find the best checkpoint in a run folder and load its model.
    
    Args:
        config: Configuration object containing experiment settings
        run_folder: Path to the run folder containing checkpoints
        data_parameters: Optional data parameters for model preparation
    
    Returns:
        tuple: (model, best_checkpoint_path, best_val_loss) or (None, None, None) if no checkpoint found
    """
    best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
    if not best_checkpoint:
        return None, None, None
    
    config.checkpoint_path = best_checkpoint
    _, _, model, _ = prepare_data_and_model(config, data_parameters)
    model.to("cuda")
    model.eval()
    return model, best_checkpoint, best_val_loss


def evaluate_best_checkpoint(config, data_parameters=None, reference_samples=None):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    if not os.path.exists(experiment_path):
        return
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    print(f'Running eval on {config.experiment_name} with {len(run_folders)} runs')
    results = {}
    
    for run_folder in run_folders:
        # only run if ds1200 or ds2400 in run foler
        # if "ds1200" not in run_folder and "ds2400" not in run_folder:
        #     continue
        print(run_folder, flush=True)
        model, best_checkpoint, best_val_loss = load_best_checkpoint_model(config, run_folder, data_parameters)
        
        if model:            
            param_scaler = data_parameters[0][0]
            
            metrics = compute_log_prob(model)
            eval_metrics = generate_samples_and_run_eval(model, param_scaler, reference_samples)
            
            results[run_folder] = {
                "best_checkpoint": best_checkpoint,
                "best_val_loss": best_val_loss,
                "metrics": {**metrics, **eval_metrics}
            }
            
            results_path = os.path.join(run_folder, "evaluation_results.json")
            with open(results_path, "w") as f:
                json.dump(results[run_folder], f, indent=4)
    
    return results

import os
import json
import numpy as np
from pathlib import Path

def parse_results(experiment_name, base_path="/share/gpu0/asaoulis/cmd/checkpoints"):
    """
    Parses evaluation results from checkpoint directories and computes mean and standard error for metrics.
    
    Args:
        experiment_name (str): The name of the experiment.
        base_path (str): The base directory where experiment folders are stored.
    
    Returns:
        dict: A dictionary where the keys are the base run names (without _i suffix) and values contain
              mean and standard error for each metric.
    """
    experiment_path = os.path.join(base_path, experiment_name)
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    
    results = {}
    aggregated_results = {}
    ensemble_results ={}
    # parse ensemble results
    ensemble_result_paths = Path(experiment_path).glob('ensemble_evaluation_results_*.json')
    for ensemble_result in ensemble_result_paths:
        # extract match_string from file name
        match_string = ensemble_result.name.split('_')[-1].split('.')[0]
        with open(ensemble_result, "r") as f:
            results_data = json.load(f)
            ensemble_results['ensemble_'+match_string] = results_data["metrics"]
    
    for run_folder in run_folders:
        results_file = os.path.join(run_folder, "evaluation_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_data = json.load(f)
                lowest_level_folder = os.path.basename(run_folder)  # Extract the last folder name
                results[lowest_level_folder] = results_data["metrics"]
    
    # Aggregate results by base run name (without _i suffix)
    for run_name, metrics in results.items():
        base_name = "_".join(run_name.split("_")[:-1]) if run_name.split("_")[-1].isdigit() else run_name
        
        if base_name not in aggregated_results:
            aggregated_results[base_name] = {}
        
        for metric, value in metrics.items():
            if metric not in aggregated_results[base_name]:
                aggregated_results[base_name][metric] = []
            aggregated_results[base_name][metric].append(value)
    
    # Compute mean and standard error for each metric
    final_results = {}
    for ensemble_name, metrics in ensemble_results.items():
        final_results[ensemble_name] = {}
        for metric, values in metrics.items():
            if metric == "fom":
                values = values
            final_results[ensemble_name][metric] = values
    for base_name, metrics in aggregated_results.items():
        final_results[base_name] = {}
        for metric, values in metrics.items():
            values = np.array(values)
            if metric == "calibration_error":
                values = values
            mean = np.mean(values)
            stderr = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error
            final_results[base_name][metric] = {"mean": mean, "stderr": stderr}
    
    return final_results

def load_best_model_and_build_posterior(config, ds_string_match="", data_parameters = None):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

    best_model_path = None
    best_val_loss = float("inf")
    best_model = None

    for run_folder in run_folders:
        if ds_string_match not in run_folder:
            continue
        model, best_model_path, val_loss = load_best_checkpoint_model(config, run_folder, data_parameters)
        if model and val_loss < best_val_loss:
            best_model = model
            best_model_path = best_model_path
            best_val_loss = val_loss
            scalers = data_parameters[0]
    if best_model is not None:
        return best_model, best_model.test_dataloader, scalers
    else:
        print("No valid checkpoints found.")
        return None
