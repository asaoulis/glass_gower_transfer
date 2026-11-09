"""
Test of Random Points (TARP) method for coverage estimation.

tarp currently implements the distance to random point (DRP) test
 introduced in Lemos, Coogan et al 2023, which relies on posterior samples.
https://github.com/Ciela-Institute/tarp/tree/main?tab=readme-ov-file
"""

from typing import Tuple, Union
import numpy as np
from tqdm import tqdm
import deprecation


__all__ = ("get_tarp_coverage", "get_drp_coverage")


def _compute_bin_membership(values: np.ndarray, alpha_edges: np.ndarray) -> dict:
    """Return per-simulation bin assignments for credibility values."""
    num_bins = len(alpha_edges) - 1
    bin_indices = np.digitize(values, alpha_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Keep arrays for direct downstream indexing into theta/samples.
    bin_members = [np.where(bin_indices == i)[0] for i in range(num_bins)]
    bin_counts = np.bincount(bin_indices, minlength=num_bins)

    return {
        "credibility_values": values,
        "bin_indices": bin_indices,
        "bin_edges": alpha_edges,
        "bin_members": bin_members,
        "bin_counts": bin_counts,
    }


@deprecation.deprecated(
    deprecated_in="0.1.0",
    removed_in="0.2.0",
    current_version="0.1.0",
    details="Use get_tarp_coverage instead",
)
def get_drp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    return get_tarp_coverage(samples=samples,
                             theta=theta,
                             references=references,
                             metric=metric,
                             num_alpha_bins=None,
                             norm=True,
                             bootstrap=False,
                             seed=None)


def _get_tarp_coverage_single(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = 100,
    norm: bool = True,
    seed: Union[int, None] = None,
    return_bin_membership: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict]]:
    """Estimates coverage with the TARP method a single time (Lemos, Coogan et al. 2023).

    samples (n_samples, n_sims, n_dims), theta (n_sims, n_dims). references are
    (n_references, n_sims) or "random" (the unit hypercube). metric is "euclidean" or
    "manhattan". num_alpha_bins=None uses n_sims // 10. Returns (ecp, alpha), plus a dict of
    per-simulation credibility values and bins with return_bin_membership.
    """
    np.random.seed(seed)

    # Check that shapes are correct
    if samples.ndim != 3:
        raise ValueError("samples must be a 3D array")

    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    if num_alpha_bins is None:
        num_alpha_bins = num_sims // 10

    if theta.shape[0] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")

    if theta.shape[1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    # Reshape theta
    theta = theta[np.newaxis, :, :]

    # Normalize
    if norm:
        low = np.min(theta, axis=1, keepdims=True)
        high = np.max(theta, axis=1, keepdims=True)
        samples = (samples - low) / (high - low + 1e-10)
        theta = (theta - low) / (high - low + 1e-10)

    # Generate reference points
    if isinstance(references, str) and references == "random":
        references = np.random.uniform(low=0, high=1, size=(num_sims, num_dims))
    else:
        assert isinstance(references, np.ndarray)  # to quiet pyright
        if references.ndim != 2:
            raise ValueError("references must be a 2D array")

        if references.shape[0] != num_sims:
            raise ValueError("references must have the same number of rows as samples")

        if references.shape[1] != num_dims:
            raise ValueError(
                "references must have the same number of columns as samples"
            )

    # Compute distances
    if metric == "euclidean":
        samples_distances = np.sqrt(
            np.sum((references[np.newaxis] - samples) ** 2, axis=-1)
        )
        theta_distances = np.sqrt(np.sum((references - theta) ** 2, axis=-1))
    elif metric == "manhattan":
        samples_distances = np.sum(np.abs(references[np.newaxis] - samples), axis=-1)
        theta_distances = np.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")

    # Compute coverage
    f = np.sum((samples_distances < theta_distances), axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins=num_alpha_bins, range=(0,1))
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h) * dx
    ecp = np.concatenate([[0], ecp])

    if return_bin_membership:
        membership = _compute_bin_membership(f, alpha)
        return ecp, alpha, membership

    return ecp, alpha


def _get_tarp_coverage_bootstrap(samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = 100,
    num_bootstrap: int = 100,
    norm: bool = True,
    seed: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap the TARP expected coverage over simulations (Lemos, Coogan et al. 2023).

    Arguments as _get_tarp_coverage_single, plus num_bootstrap resamples. Returns
    (ecp of shape (num_bootstrap, num_alpha_bins + 1), alpha).
    """
    num_sims = samples.shape[1]

    if num_alpha_bins is None:
        num_alpha_bins = num_sims // 10

    boot_ecp = np.empty(shape=(num_bootstrap, num_alpha_bins+1))
    for i in tqdm(range(num_bootstrap)):
        idx = np.random.randint(low=0, high=num_sims, size=num_sims)
        
        # Sample with replacement from the full set of simulations
        boot_samples = samples[:, idx, :]
        boot_theta = theta[idx, :]

        boot_ecp[i, :], alpha = _get_tarp_coverage_single(boot_samples,
                                                          boot_theta,
                                                          references=references,
                                                          metric=metric,
                                                          num_alpha_bins=num_alpha_bins,
                                                          norm=norm,
                                                          seed=seed)

    return boot_ecp, alpha


def get_tarp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
    num_alpha_bins: Union[int, None] = None,
    num_bootstrap: int = 100,
    norm: bool = False,
    bootstrap: bool = False,
    seed: Union[int, None] = None,
    return_bin_membership: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict]]:
    """Estimates coverage with the TARP method (Lemos, Coogan et al. 2023).

    samples (n_samples, n_sims, n_dims), theta (n_sims, n_dims). Returns (ecp, alpha). With
    bootstrap, ecp gains a leading bootstrap axis. With return_bin_membership, also returns a dict
    of per-simulation bin assignments from one non-bootstrap pass.
    """
    if bootstrap:
        ecp, alpha = _get_tarp_coverage_bootstrap(samples, theta, references, metric, num_alpha_bins, num_bootstrap,
                                                  norm, seed)
        if return_bin_membership:
            _, _, membership = _get_tarp_coverage_single(
                samples,
                theta,
                references=references,
                metric=metric,
                num_alpha_bins=num_alpha_bins,
                norm=norm,
                seed=seed,
                return_bin_membership=True,
            )
            return ecp, alpha, membership
    else:
        if return_bin_membership:
            ecp, alpha, membership = _get_tarp_coverage_single(
                samples,
                theta,
                references,
                metric,
                num_alpha_bins,
                norm,
                seed,
                return_bin_membership=True,
            )
            return ecp, alpha, membership

        ecp, alpha = _get_tarp_coverage_single(samples, theta, references, metric, num_alpha_bins, norm, seed)
    return ecp, alpha
