
import os
import torch
import glob
import re
import json

import torch
from torch.distributions import Distribution, Uniform
from sbi.utils import MultipleIndependent

from .fom import compute_fom, compute_cov_matrix_per_sim
from .tarp import get_tarp_coverage


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


def run_evaluation_on_samples(theta0s, samples, param_scaler, reference_samples=None, compute_calibration=True):
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