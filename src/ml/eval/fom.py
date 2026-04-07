import torch


def compute_cov_matrix_per_sim(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes a covariance matrix per posterior inference (per observation/simulation).

    Expected input shape (consistent with evaluate_models.py):
        X: (n_sims, n_samples, n_dims)

    Returns:
        cov_matrices: (n_sims, n_dims, n_dims)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_sims, n_samples, n_dims), got {tuple(X.shape)}")

    n_sims, n_samples, n_dims = X.shape

    # Mean over samples for each sim
    mean = X.mean(dim=1, keepdim=True)  # (n_sims, 1, n_dims)
    X_centered = X - mean               # (n_sims, n_samples, n_dims)

    # Cov per sim: (X^T X)/(n_samples-1)
    cov_matrices = torch.einsum("snd,sne->sde", X_centered, X_centered) / max(n_samples - 1, 1)

    # Numerical stabilizer (helps near-singular covariance)
    cov_matrices = cov_matrices + eps * torch.eye(n_dims, device=X.device, dtype=X.dtype).unsqueeze(0)
    return cov_matrices


def compute_fom(samples: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Dimension-normalized FoM:
        FoM_d = det(C)^(-1 / (2d))
    which is comparable across dimensions.
    """
    cov = compute_cov_matrix_per_sim(samples, eps=eps)  # (n_sims, d, d)
    d = cov.shape[-1]

    sign, logabsdet = torch.linalg.slogdet(cov)
    # Guard against numerical issues / non-PD estimates
    logabsdet = torch.where(sign > 0, logabsdet, torch.log(torch.full_like(logabsdet, eps)))

    fom_per_sim = torch.exp(-0.5 * logabsdet / d)
    return float(fom_per_sim.mean().item())