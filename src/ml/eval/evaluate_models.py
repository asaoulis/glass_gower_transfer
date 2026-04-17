import torch

import numpy as np


from .fom import (
    compute_cov_matrix_per_sim,
    compute_dim_normalized_fom,
    compute_dim_normalized_fom_against_prior,
    compute_standard_fom,
    compute_standard_fom_against_prior,
)
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


def _split_samples_first_and_sims_first(samples_t: torch.Tensor, n_sims: int, *, name: str = "samples"):
    """Return both common posterior-sample layouts.

    Many samplers (incl. SBI) produce samples shaped (n_samples, n_sims, d).
    Some code expects (n_sims, n_samples, d). This helper returns:
      - samples_first: (n_samples, n_sims, d)
      - sims_first:    (n_sims, n_samples, d)
    """
    if samples_t.ndim != 3:
        raise ValueError(f"{name} must have 3 dims (got {tuple(samples_t.shape)})")

    if samples_t.shape[0] == n_sims:
        sims_first = samples_t
        samples_first = samples_t.permute(1, 0, 2)
    elif samples_t.shape[1] == n_sims:
        samples_first = samples_t
        sims_first = samples_t.permute(1, 0, 2)
    else:
        raise ValueError(
            f"{name} has shape {tuple(samples_t.shape)} which cannot be aligned to n_sims={n_sims}. "
            "Expected either (n_sims, n_samples, d) or (n_samples, n_sims, d)."
        )

    return samples_first, sims_first


def _subset_indices(all_params, subset):
    """Return list of indices for `subset` if all exist in `all_params`, else None."""
    if all(p in all_params for p in subset):
        return [all_params.index(p) for p in subset]
    return None


def _sample_from_prior(prior, num_samples: int, target_dim: int):
    if prior is None:
        return None

    try:
        prior_samples = prior.sample((num_samples,))
    except TypeError:
        prior_samples = prior.sample(num_samples)

    if isinstance(prior_samples, (tuple, list)):
        prior_samples = prior_samples[0]

    if not torch.is_tensor(prior_samples):
        prior_samples = torch.as_tensor(prior_samples)

    if prior_samples.ndim == 1:
        prior_samples = prior_samples.unsqueeze(0)
    elif prior_samples.ndim > 2:
        prior_samples = prior_samples.reshape(-1, prior_samples.shape[-1])

    if prior_samples.shape[-1] != target_dim:
        raise ValueError(
            f"Prior samples have last dim {prior_samples.shape[-1]} but expected {target_dim}."
        )

    return prior_samples.detach().to(dtype=torch.float32)


def _cov_from_prior_samples(prior_samples_t: torch.Tensor, idx=None):
    if idx is None:
        subset = prior_samples_t
    else:
        subset = prior_samples_t[:, idx]

    if subset.ndim == 1:
        subset = subset.unsqueeze(-1)

    cov = compute_cov_matrix_per_sim(subset.unsqueeze(0))
    return cov[0]


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

    def _summarize_coverage(self, coverage):
        summary = {
            "calibration_error": self._calibration_error_from_coverage(coverage),
            "credible_intervals": coverage[1],
        }
        if self.bootstrap:
            summary["ecp_bootstrap"] = coverage[0]
        else:
            summary["ecp"] = coverage[0]
        return summary

    def _run_tarp(self, samples_t: torch.Tensor, theta_t: torch.Tensor):
        samples_np, theta_np = self._to_tarp_shapes(samples_t, theta_t)
        coverage = get_tarp_coverage(
            samples_np,
            theta_np,
            bootstrap=self.bootstrap,
            num_bootstrap=self.num_bootstrap,
            seed=self.seed,
        )
        return self._summarize_coverage(coverage)

    def compute_all(self, samples_t: torch.Tensor, theta_t: torch.Tensor):
        out = {"tarp": {"full": {}, "per_param": {}, "subsets": {}}}

        out["tarp"]["full"] = self._run_tarp(samples_t, theta_t)

        for dim, name in enumerate(self.cosmological_params):
            out["tarp"]["per_param"][name] = self._run_tarp(
                samples_t[:, :, [dim]], theta_t[:, [dim]]
            )

        subset = ["sigma_8", "omega_m", "w0"]
        idx = _subset_indices(self.cosmological_params, subset)
        if idx is not None:
            key = "__".join(subset)
            out["tarp"]["subsets"][key] = self._run_tarp(
                samples_t[:, :, idx], theta_t[:, idx]
            )

        return out


class DimNormalizedFoMDiagnostics:
    """
    Compute FoM on:
      - full posterior
      - selected subsets (if the parameters exist)
    """

    def __init__(self, cosmological_params, prior_samples_t: torch.Tensor = None):
        self.cosmological_params = list(cosmological_params)
        self.prior_samples_t = prior_samples_t

    def _compute_fom(self, samples_t: torch.Tensor, idx=None):
        if self.prior_samples_t is None:
            return compute_dim_normalized_fom(samples_t)

        prior_cov = _cov_from_prior_samples(self.prior_samples_t, idx=idx)
        return compute_dim_normalized_fom_against_prior(samples_t, prior_cov)

    def compute_full(self, samples_t: torch.Tensor):
        return {
            "fom_dim_normalized": self._compute_fom(samples_t),
            "fom_dim_normalized_uses_prior": self.prior_samples_t is not None,
        }

    def compute_per_param(self, samples_t: torch.Tensor):
        out = {"fom_dim_normalized_per_param": {}}
        for dim, name in enumerate(self.cosmological_params):
            out["fom_dim_normalized_per_param"][name] = self._compute_fom(
                samples_t[:, :, [dim]], idx=[dim]
            )
        return out

    def compute_subsets(self, samples_t: torch.Tensor):
        out = {"fom_dim_normalized_subsets": {}}

        subsets = [
            ("omega_m__sigma_8", ["omega_m", "sigma_8"]),
            ("omega_m__sigma_8__w0", ["omega_m", "sigma_8", "w0"]),
            ("a_ia__b_ia", ["a_ia", "b_ia"]),
        ]

        for key, subset in subsets:
            idx = _subset_indices(self.cosmological_params, subset)
            if idx is None:
                continue
            out["fom_dim_normalized_subsets"][key] = self._compute_fom(samples_t[:, :, idx], idx=idx)

        return out

    def compute_all(self, samples_t: torch.Tensor):
        out = {}
        out.update(self.compute_full(samples_t))
        out.update(self.compute_per_param(samples_t))
        out.update(self.compute_subsets(samples_t))
        return out


class StandardFoMDiagnostics:
    """
    Compute standard FoM (det(C)^(-1/2)) on:
      - full posterior
      - selected subsets (if the parameters exist)
      - selected subsets involving derived S8 (if required parameters exist)
    """

    def __init__(self, cosmological_params, prior_samples_t: torch.Tensor = None):
        self.cosmological_params = list(cosmological_params)
        self.prior_samples_t = prior_samples_t

    def _compute_fom(self, samples_t: torch.Tensor, idx=None):
        if self.prior_samples_t is None:
            return compute_standard_fom(samples_t)

        prior_cov = _cov_from_prior_samples(self.prior_samples_t, idx=idx)
        return compute_standard_fom_against_prior(samples_t, prior_cov)

    def compute_full(self, samples_t: torch.Tensor):
        return {
            "fom": self._compute_fom(samples_t),
            "fom_uses_prior": self.prior_samples_t is not None,
        }

    def compute_subsets(self, samples_t: torch.Tensor):
        out = {"fom_subsets": {}}

        subsets = [
            ("omega_m__sigma_8", ["omega_m", "sigma_8"]),
            ("omega_m__w0", ["omega_m", "w0"]),
        ]

        for key, subset in subsets:
            idx = _subset_indices(self.cosmological_params, subset)
            if idx is None:
                continue
            out["fom_subsets"][key] = self._compute_fom(samples_t[:, :, idx], idx=idx)

        idx_s8 = _subset_indices(self.cosmological_params, ["sigma_8", "omega_m", "w0"])
        if idx_s8 is not None:
            i_sigma8 = self.cosmological_params.index("sigma_8")
            i_omegam = self.cosmological_params.index("omega_m")
            i_w0 = self.cosmological_params.index("w0")

            s8_samples = (
                samples_t[:, :, i_sigma8]
                * (samples_t[:, :, i_omegam] / 0.3) ** 0.5
            )
            s8_w0_samples = torch.stack((s8_samples, samples_t[:, :, i_w0]), dim=-1)
            if self.prior_samples_t is not None:
                s8_prior = (
                    self.prior_samples_t[:, i_sigma8]
                    * (self.prior_samples_t[:, i_omegam] / 0.3) ** 0.5
                )
                s8_w0_prior = torch.stack((s8_prior, self.prior_samples_t[:, i_w0]), dim=-1)
                prior_cov = _cov_from_prior_samples(s8_w0_prior)
                out["fom_subsets"]["s8__w0"] = compute_standard_fom_against_prior(
                    s8_w0_samples,
                    prior_cov,
                )
            else:
                out["fom_subsets"]["s8__w0"] = compute_standard_fom(s8_w0_samples)

        return out

    def compute_all(self, samples_t: torch.Tensor):
        out = {}
        out.update(self.compute_full(samples_t))
        out.update(self.compute_subsets(samples_t))
        return out


def run_evaluation_on_samples(
    theta0s,
    samples,
    param_scaler,
    reference_samples=None,
    compute_calibration=True,
    prior=None,
    prior_num_samples=20_000,
):
    cosmological_params = param_scaler.parameter_names
    n_sims = theta0s.shape[0]

    scaled_theta0s = rescale_parameters(theta0s, param_scaler)
    scaled_samples = rescale_parameters(samples, param_scaler)

    # Keep both layouts around. Most point-estimate metrics below assume
    # (n_samples, n_sims, d), while covariance/FoM metrics require
    # (n_sims, n_samples, d).
    scaled_samples_first, scaled_samples_sims_first = _split_samples_first_and_sims_first(
        scaled_samples, n_sims, name="scaled_samples"
    )
    _, samples_sims_first = _split_samples_first_and_sims_first(samples, n_sims, name="samples")

    sample_means = scaled_samples_first.mean(axis=0)
    mse = torch.nn.functional.mse_loss(sample_means, scaled_theta0s, reduction='none')
    bias = scaled_samples_first.mean(axis=0) - scaled_theta0s

    std_devs = scaled_samples_first.std(axis=0)
    width_68 = torch.quantile(scaled_samples_first, 0.84, dim=0) - torch.quantile(scaled_samples_first, 0.16, dim=0)
    width_95 = torch.quantile(scaled_samples_first, 0.975, dim=0) - torch.quantile(scaled_samples_first, 0.025, dim=0)

    prior_samples_scaled = _sample_from_prior(prior, prior_num_samples, target_dim=samples.shape[-1])
    prior_samples_unscaled = (
        rescale_parameters(prior_samples_scaled, param_scaler)
        if prior_samples_scaled is not None
        else None
    )

    eval_metrics = {
        "sample_ensemble_mse": mse.mean().item(),
    }

    # Keep previous custom behavior under a more explicit metric name.
    eval_metrics.update(
        DimNormalizedFoMDiagnostics(
            cosmological_params,
            prior_samples_t=prior_samples_scaled,
        ).compute_all(samples_sims_first)
    )

    # Save standard FoM using the canonical FoM key and requested subsets.
    eval_metrics.update(
        StandardFoMDiagnostics(
            cosmological_params,
            prior_samples_t=prior_samples_unscaled,
        ).compute_all(scaled_samples_sims_first)
    )

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
            scaled_samples_first[:, :, i_sigma8]
            * (scaled_samples_first[:, :, i_omegam] / 0.3) ** 0.5
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

    if bias.shape[0] != n_sims:
        raise ValueError(
            f"Unexpected bias shape {tuple(bias.shape)}; expected first dim n_sims={n_sims}."
        )

    cov_matrices = compute_cov_matrix_per_sim(scaled_samples_sims_first)  # (n_sims, d, d)
    inv_covariances = torch.linalg.inv(cov_matrices)

    mahalanobis_distances = torch.sqrt(torch.einsum("bi,bij,bj->b", bias, inv_covariances, bias))
    eval_metrics['mahalanobis_distance_mean'] = mahalanobis_distances.mean().item()
    eval_metrics['mahalanobis_distance_std'] = mahalanobis_distances.std().item()

    if compute_calibration:
        # TARP diagnostics operate in scaled space; pass sims-first samples.
        tarp = TARPDiagnostics(
            cosmological_params=cosmological_params,
            bootstrap=True,
            num_bootstrap=25,
            seed=None,
        )
        eval_metrics.update(tarp.compute_all(samples_sims_first, theta0s))

    if reference_samples is not None:
        reference_samples_t = torch.as_tensor(reference_samples)
        scaled_reference_samples = rescale_parameters(reference_samples_t, param_scaler)

        ref_first, ref_sims_first = _split_samples_first_and_sims_first(
            scaled_reference_samples, n_sims, name="scaled_reference_samples"
        )

        # Match number of posterior samples along the sampling axis.
        n_match = min(scaled_samples_first.shape[0], ref_first.shape[0])
        post_first = scaled_samples_first[:n_match]
        ref_first = ref_first[:n_match]
        post_sims_first = scaled_samples_sims_first[:, :n_match]
        ref_sims_first = ref_sims_first[:, :n_match]

        eval_metrics.update({
            "ref_post_mean_mse": torch.nn.functional.mse_loss(
                post_first.mean(axis=0), ref_first.mean(axis=0)
            ).item(),
            "ref_post_cov_mse": torch.nn.functional.mse_loss(
                compute_cov_matrix_per_sim(post_sims_first), compute_cov_matrix_per_sim(ref_sims_first)
            ).item()
        })

    return eval_metrics