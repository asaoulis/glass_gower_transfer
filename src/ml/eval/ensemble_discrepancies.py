"""Ensemble agreement / discrepancy diagnostics.

This module centralizes the metrics used by `ensemble_uncertainty.py` so that
scripts can stay thin and evaluation logic is reusable.

Currently includes:
- RBF-kernel agreement across ensemble members (MMD-style) for intermediate
  vectors like representations or compressed latents.
- Symmetric KL disagreement between diagonal-Gaussian approximations fitted to
  posterior samples.
"""

from __future__ import annotations

import numpy as np


def median_heuristic_sigma2(vectors: np.ndarray, *, max_pairs: int = 20_000, seed: int = 0) -> float:
    """Median heuristic for the RBF bandwidth parameter $\sigma^2$.

    Parameters
    ----------
    vectors:
        Array with shape [M, D].
    max_pairs:
        Maximum number of randomly sampled pairs used to estimate the median.
    seed:
        RNG seed for pair subsampling.

    Returns
    -------
    sigma2:
        Positive float bandwidth parameter. Falls back to 1.0 in degenerate cases.
    """

    rng = np.random.default_rng(seed)
    m = int(vectors.shape[0])
    if m < 2:
        return 1.0

    n_pairs = min(int(max_pairs), m * (m - 1) // 2)

    i = rng.integers(0, m, size=n_pairs)
    j = rng.integers(0, m, size=n_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    if i.size == 0:
        return 1.0

    diffs = vectors[i] - vectors[j]
    d2 = np.sum(diffs * diffs, axis=-1)
    med = float(np.median(d2))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return med


def rbf_kernel_avg_pair(vectors: np.ndarray, sigma2: float) -> np.ndarray:
    """Per-example average RBF kernel value across unordered member pairs.

    Parameters
    ----------
    vectors:
        Array with shape [N, K, D].
    sigma2:
        RBF bandwidth parameter $\sigma^2$.

    Returns
    -------
    avg_k:
        Array with shape [N], where each entry is the mean of
        $\exp(-\|x_i-x_j\|^2/(2\sigma^2))$ over all i<j.
    """

    n, k, _d = vectors.shape
    if k < 2:
        return np.zeros(n, dtype=np.float64)

    a2 = np.sum(vectors * vectors, axis=-1, keepdims=True)  # [N,K,1]
    dot = np.matmul(vectors, np.transpose(vectors, (0, 2, 1)))  # [N,K,K]
    d2 = a2 + np.transpose(a2, (0, 2, 1)) - 2.0 * dot
    d2 = np.maximum(d2, 0.0)

    iu = np.triu_indices(k, k=1)
    d2_u = d2[:, iu[0], iu[1]]
    k_u = np.exp(-d2_u / (2.0 * float(sigma2)))
    return k_u.mean(axis=-1)


def mmd_disagreement_score(vectors: np.ndarray, sigma2: float) -> np.ndarray:
    """MMD-style per-example disagreement score in [0,1] (higher = more disagree).

    Defined as:
        1 - mean_{i<j} k(x_i, x_j)
    with an RBF kernel k.

    Parameters
    ----------
    vectors:
        Array with shape [N, K, D].
    sigma2:
        RBF bandwidth parameter $\sigma^2$.

    Returns
    -------
    score:
        Array with shape [N].
    """

    avg_k = rbf_kernel_avg_pair(vectors, sigma2=sigma2)
    return 1.0 - avg_k


def diag_gaussian_symmetric_kl(mu: np.ndarray, var: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Mean symmetric KL across ensemble members per example.

    Parameters
    ----------
    mu:
        Means with shape [K, N, D].
    var:
        Diagonal variances with shape [K, N, D].
    eps:
        Minimum variance clamp for numerical stability.

    Returns
    -------
    scores:
        Array with shape [N], averaged over all unordered member pairs.
    """

    k, n, _d = mu.shape
    if k < 2:
        return np.zeros(n, dtype=np.float64)

    var = np.maximum(var, eps)

    scores = np.zeros(n, dtype=np.float64)
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            mu_i = mu[i]
            mu_j = mu[j]
            vi = var[i]
            vj = var[j]

            kl_ij = 0.5 * np.sum(
                np.log(vj / vi) + (vi + (mu_i - mu_j) ** 2) / vj - 1.0,
                axis=-1,
            )
            kl_ji = 0.5 * np.sum(
                np.log(vi / vj) + (vj + (mu_j - mu_i) ** 2) / vi - 1.0,
                axis=-1,
            )

            scores += 0.5 * (kl_ij + kl_ji)
            count += 1

    return scores / float(count)
