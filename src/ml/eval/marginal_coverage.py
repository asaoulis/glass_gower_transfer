"""
Alternative 1-D marginal coverage estimators for posterior calibration checks.

This module provides two estimators that can be compared against TARP:
1) Quantile-based estimator using posterior CDF values at the true parameter.
2) KDE-based estimator using 1-D HPD mass at the true parameter density level.

Both return outputs compatible with the TARP-style summary:
- expected coverage probability (ecp)
- credibility interval grid (alpha)
- optional bootstrap draws of ecp
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .tarp import get_tarp_coverage

try:
    from scipy.stats import gaussian_kde
except ImportError as exc:  # pragma: no cover
    gaussian_kde = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


__all__ = (
    "get_qq_1d_coverage",
    "get_kde_1d_coverage",
    "summarize_coverage",
    "compute_1d_coverage_diagnostics",
)


def _ensure_1d_inputs(samples: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples)
    theta = np.asarray(theta)

    if samples.ndim == 3:
        if samples.shape[-1] != 1:
            raise ValueError("For 1-D coverage, samples last dimension must be 1.")
        samples = samples[:, :, 0]
    elif samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_sims) or (n_samples, n_sims, 1).")

    if theta.ndim == 2:
        if theta.shape[-1] != 1:
            raise ValueError("For 1-D coverage, theta last dimension must be 1.")
        theta = theta[:, 0]
    elif theta.ndim != 1:
        raise ValueError("theta must have shape (n_sims,) or (n_sims, 1).")

    if samples.shape[1] != theta.shape[0]:
        raise ValueError("samples and theta have incompatible n_sims dimensions.")

    return samples, theta


def _default_num_alpha_bins(n_sims: int) -> int:
    return max(1, n_sims // 10)


def _coverage_from_values(values: np.ndarray, num_alpha_bins: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    n_sims = values.shape[0]
    if num_alpha_bins is None:
        num_alpha_bins = _default_num_alpha_bins(n_sims)

    hist, alpha = np.histogram(values, density=True, bins=num_alpha_bins, range=(0.0, 1.0))
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(hist) * dx
    return np.concatenate([[0.0], ecp]), alpha


def _bootstrap_coverage(
    values: np.ndarray,
    num_alpha_bins: Optional[int],
    num_bootstrap: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    n_sims = values.shape[0]
    if num_alpha_bins is None:
        num_alpha_bins = _default_num_alpha_bins(n_sims)

    rng = np.random.default_rng(seed)
    boot_ecp = np.empty((num_bootstrap, num_alpha_bins + 1), dtype=float)

    alpha_out = None
    for i in range(num_bootstrap):
        idx = rng.integers(low=0, high=n_sims, size=n_sims)
        boot_ecp[i], alpha = _coverage_from_values(values[idx], num_alpha_bins)
        alpha_out = alpha

    if alpha_out is None:
        raise RuntimeError("Bootstrap failed to produce alpha grid.")

    return boot_ecp, alpha_out


def _calibration_error_from_ecp(ecp: np.ndarray) -> float:
    mean_ecp = ecp if ecp.ndim == 1 else ecp.mean(axis=0)
    rank_hist = np.diff(mean_ecp)
    rank_hist *= len(rank_hist)
    expected = np.ones(len(rank_hist))
    return float(np.mean((rank_hist - expected) ** 2))


def summarize_coverage(coverage: Tuple[np.ndarray, np.ndarray], bootstrap: bool) -> Dict[str, Any]:
    ecp, alpha = coverage
    out: Dict[str, Any] = {
        "calibration_error": _calibration_error_from_ecp(ecp),
        "credible_intervals": alpha,
    }
    if bootstrap:
        out["ecp_bootstrap"] = ecp
    else:
        out["ecp"] = ecp
    return out


def get_qq_1d_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    num_alpha_bins: Optional[int] = None,
    bootstrap: bool = False,
    num_bootstrap: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1-D quantile-quantile style coverage estimator.

    For each simulation i, computes u_i = P(theta <= theta_true_i | x_i)
    from posterior samples. For a calibrated posterior, u_i should be Uniform(0,1).
    Coverage is then estimated from the empirical CDF of {u_i}.
    """
    samples_1d, theta_1d = _ensure_1d_inputs(samples, theta)
    u = np.mean(samples_1d <= theta_1d[np.newaxis, :], axis=0)

    if bootstrap:
        return _bootstrap_coverage(u, num_alpha_bins, num_bootstrap, seed)
    return _coverage_from_values(u, num_alpha_bins)


def _kde_hpd_mass_1d(
    sample_vec: np.ndarray,
    theta_true: float,
    grid_size: int,
    grid_padding_std: float,
) -> float:
    if gaussian_kde is None:  # pragma: no cover
        raise ImportError("scipy is required for KDE coverage.") from _SCIPY_IMPORT_ERROR

    std = float(np.std(sample_vec))
    if std <= 0.0:
        return 1.0 if np.isclose(theta_true, sample_vec[0]) else 0.0

    # Small jitter makes gaussian_kde robust to near-singular sample variance.
    jittered = sample_vec + np.random.normal(scale=1e-8 * std, size=sample_vec.shape)
    kde = gaussian_kde(jittered)

    lower = min(np.min(sample_vec), theta_true) - grid_padding_std * std
    upper = max(np.max(sample_vec), theta_true) + grid_padding_std * std
    grid = np.linspace(lower, upper, grid_size)

    pdf_grid = kde(grid)
    dx = grid[1] - grid[0]
    theta_pdf = float(kde([theta_true])[0])

    mass_superlevel = float(np.sum(pdf_grid[pdf_grid >= theta_pdf]) * dx)
    total_mass = float(np.sum(pdf_grid) * dx)

    if total_mass <= 0.0:
        return 0.0
    return float(np.clip(mass_superlevel / total_mass, 0.0, 1.0))


def get_kde_1d_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    num_alpha_bins: Optional[int] = None,
    bootstrap: bool = False,
    num_bootstrap: int = 100,
    seed: Optional[int] = None,
    grid_size: int = 512,
    grid_padding_std: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1-D KDE-based HPD coverage estimator.

    For each simulation i:
    1) Fit a 1-D KDE to posterior samples.
    2) Evaluate p(theta_true_i).
    3) Compute the HPD mass f_i = int_{p(x)>=p(theta_true_i)} p(x) dx.

    For calibrated posteriors, f_i should be Uniform(0,1), and coverage is estimated
    from the empirical CDF of {f_i}.
    """
    samples_1d, theta_1d = _ensure_1d_inputs(samples, theta)

    f = np.empty(theta_1d.shape[0], dtype=float)
    for i in range(theta_1d.shape[0]):
        f[i] = _kde_hpd_mass_1d(
            sample_vec=samples_1d[:, i],
            theta_true=float(theta_1d[i]),
            grid_size=grid_size,
            grid_padding_std=grid_padding_std,
        )

    if bootstrap:
        return _bootstrap_coverage(f, num_alpha_bins, num_bootstrap, seed)
    return _coverage_from_values(f, num_alpha_bins)


def compute_1d_coverage_diagnostics(
    samples: np.ndarray,
    theta: np.ndarray,
    *,
    bootstrap: bool = True,
    num_bootstrap: int = 100,
    num_alpha_bins: Optional[int] = None,
    seed: Optional[int] = None,
    tarp_kwargs: Optional[Dict[str, Any]] = None,
    kde_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run TARP, 1-D quantile, and 1-D KDE coverage diagnostics on 1-D marginals.

    Inputs can be either:
    - samples: (n_samples, n_sims, 1) or (n_samples, n_sims)
    - theta: (n_sims, 1) or (n_sims,)
    """
    samples_1d, theta_1d = _ensure_1d_inputs(samples, theta)

    samples_tarp = samples_1d[:, :, np.newaxis]
    theta_tarp = theta_1d[:, np.newaxis]

    tarp_opts = dict(tarp_kwargs or {})
    kde_opts = dict(kde_kwargs or {})

    tarp_cov = get_tarp_coverage(
        samples=samples_tarp,
        theta=theta_tarp,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap,
        num_alpha_bins=num_alpha_bins,
        seed=seed,
        **tarp_opts,
    )
    qq_cov = get_qq_1d_coverage(
        samples=samples_1d,
        theta=theta_1d,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap,
        num_alpha_bins=num_alpha_bins,
        seed=seed,
    )
    kde_cov = get_kde_1d_coverage(
        samples=samples_1d,
        theta=theta_1d,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap,
        num_alpha_bins=num_alpha_bins,
        seed=seed,
        **kde_opts,
    )

    return {
        "tarp": summarize_coverage(tarp_cov, bootstrap=bootstrap),
        "qq_1d": summarize_coverage(qq_cov, bootstrap=bootstrap),
        "kde_1d": summarize_coverage(kde_cov, bootstrap=bootstrap),
    }
