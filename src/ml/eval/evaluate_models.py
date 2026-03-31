import torch

import numpy as np


from .fom import compute_fom, compute_cov_matrix_per_sim
from .tarp import get_tarp_coverage


def rescale_parameters(tensor, scaler):
    """
    Rescales the given tensor using the provided scaler, ensuring the correct shape.
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    original_shape = tensor.shape
    reshaped_tensor = tensor.reshape(-1, original_shape[-1])  # Flatten to (N, d) for scaling
    scaled_array = scaler.inverse_transform(reshaped_tensor.cpu().numpy())  # Apply inverse scaling
    scaled_tensor = torch.tensor(scaled_array, device=(device), dtype=torch.float32)  # Convert back to tensor
    return scaled_tensor.reshape(original_shape)  # Restore original shape


def _subset_indices(all_params, subset):
    """Return list of indices for `subset` if all exist in `all_params`, else None."""
    if all(p in all_params for p in subset):
        return [all_params.index(p) for p in subset]
    return None


class TARPDiagnostics:
    """
    Helper to compute TARP-based calibration metrics.
    """
    def __init__(self, cosmological_params, bootstrap=True, num_bootstrap=25, seed=None):
        self.cosmological_params = list(cosmological_params)
        self.bootstrap = bootstrap
        self.num_bootstrap = num_bootstrap
        self.seed = seed

    @staticmethod
    def _to_tarp_shapes(samples_t: torch.Tensor, theta_t: torch.Tensor):
        samples_np = samples_t.detach().cpu().permute(1, 0, 2).numpy()
        theta_np = theta_t.detach().cpu().numpy()
        return samples_np, theta_np

    @staticmethod
    def _calibration_error_from_coverage(coverage):
        ecp = coverage[0]
        mean_ecp = ecp if ecp.ndim == 1 else ecp.mean(axis=0)
        rank_hist = np.diff(mean_ecp)
        rank_hist *= len(rank_hist)
        expected = np.ones(len(rank_hist))
        return float(np.mean((rank_hist - expected) ** 2))

    def _run_tarp(self, samples_t: torch.Tensor, theta_t: torch.Tensor):
        samples_np, theta_np = self._to_tarp_shapes(samples_t, theta_t)
        coverage = get_tarp_coverage(
            samples_np,
            theta_np,
            bootstrap=self.bootstrap,
            num_bootstrap=self.num_bootstrap,
            seed=self.seed,
        )
        return self._calibration_error_from_coverage(coverage)

    def compute_all(self, samples_t: torch.Tensor, theta_t: torch.Tensor):
        out = {"tarp": {"full": {}, "per_param": {}, "subsets": {}}}

        out["tarp"]["full"]["calibration_error"] = self._run_tarp(samples_t, theta_t)

        for dim, name in enumerate(self.cosmological_params):
            ce = self._run_tarp(samples_t[:, :, [dim]], theta_t[:, [dim]])
            out["tarp"]["per_param"][name] = {"calibration_error": ce}

        subset = ["sigma_8", "omega_m", "w0"]
        idx = _subset_indices(self.cosmological_params, subset)
        if idx is not None:
            key = "__".join(subset)
            out["tarp"]["subsets"][key] = {
                "calibration_error": self._run_tarp(samples_t[:, :, idx], theta_t[:, idx])
            }

        return out


class FoMDiagnostics:
    """
    Compute FoM on:
      - full posterior
      - selected subsets (if the parameters exist)
    """

    def __init__(self, cosmological_params):
        self.cosmological_params = list(cosmological_params)

    def compute_full(self, samples_t: torch.Tensor):
        return {"fom": compute_fom(samples_t)}

    def compute_subsets(self, samples_t: torch.Tensor):
        out = {"fom_subsets": {}}

        subsets = [
            ("omega_m__sigma_8", ["omega_m", "sigma_8"]),
            ("omega_m__sigma_8__w0", ["omega_m", "sigma_8", "w0"]),
            ("a_ia__b_ia", ["a_ia", "b_ia"]),
        ]

        for key, subset in subsets:
            idx = _subset_indices(self.cosmological_params, subset)
            if idx is None:
                continue
            out["fom_subsets"][key] = compute_fom(samples_t[:, :, idx])

        return out

    def compute_all(self, samples_t: torch.Tensor):
        out = {}
        out.update(self.compute_full(samples_t))
        out.update(self.compute_subsets(samples_t))
        return out


def run_evaluation_on_samples(theta0s, samples, param_scaler, reference_samples=None, compute_calibration=True):
    cosmological_params = param_scaler.parameter_names
    scaled_theta0s = rescale_parameters(theta0s, param_scaler)
    scaled_samples = rescale_parameters(samples, param_scaler)

    sample_means = scaled_samples.mean(axis=0)
    mse = torch.nn.functional.mse_loss(sample_means, scaled_theta0s, reduction='none')
    bias = scaled_samples.mean(axis=0) - scaled_theta0s

    std_devs = scaled_samples.std(axis=0)
    width_68 = torch.quantile(scaled_samples, 0.84, dim=0) - torch.quantile(scaled_samples, 0.16, dim=0)
    width_95 = torch.quantile(scaled_samples, 0.975, dim=0) - torch.quantile(scaled_samples, 0.025, dim=0)

    eval_metrics = {
        # FoM should generally be computed on the same "space" you care about.
        # Keeping your existing behavior: unscaled `samples`.
        "fom": compute_fom(samples),
        "sample_ensemble_mse": mse.mean().item(),
    }

    # Add FoM subset diagnostics (unscaled, consistent with eval_metrics["fom"])
    eval_metrics.update(FoMDiagnostics(cosmological_params).compute_subsets(samples))

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
        )
        s8_theta0s = (
            scaled_theta0s[:, i_sigma8]
            * (scaled_theta0s[:, i_omegam] / 0.3) ** 0.5
        )

        s8_mean = s8_samples.mean(dim=0)

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

    n_sims = theta0s.shape[0]
    if bias.shape[0] != n_sims:
        raise ValueError(
            f"Unexpected bias shape {tuple(bias.shape)}; expected first dim n_sims={n_sims}."
        )

    if scaled_samples.shape[0] == n_sims:
        samples_for_cov = scaled_samples
    elif scaled_samples.shape[1] == n_sims:
        samples_for_cov = scaled_samples.permute(1, 0, 2)
    else:
        raise ValueError(
            f"scaled_samples has shape {tuple(scaled_samples.shape)} which cannot be aligned to n_sims={n_sims}"
        )

    cov_matrices = compute_cov_matrix_per_sim(samples_for_cov)  # (n_sims, d, d)
    inv_covariances = torch.linalg.inv(cov_matrices)

    mahalanobis_distances = torch.sqrt(torch.einsum("bi,bij,bj->b", bias, inv_covariances, bias))
    eval_metrics['mahalanobis_distance_mean'] = mahalanobis_distances.mean().item()
    eval_metrics['mahalanobis_distance_std'] = mahalanobis_distances.std().item()

    if compute_calibration:
        tarp = TARPDiagnostics(
            cosmological_params=cosmological_params,
            bootstrap=True,
            num_bootstrap=25,
            seed=None,
        )
        eval_metrics.update(tarp.compute_all(samples, theta0s))

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