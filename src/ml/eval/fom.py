import torch
def compute_cov_matrix_per_sim(X):
    """
    Computes the covariance matrix per simulation.

    Args:
        X (torch.Tensor): Input tensor of shape (num_samples, num_sims, dim)

    Returns:
        torch.Tensor: Covariance matrices of shape (num_sims, dim, dim)
    """
    num_samples, num_sims, dim = X.shape
    
    # Compute mean along the sample dimension
    mean = X.mean(dim=0, keepdim=True)  # Shape: (1, num_sims, dim)
    
    # Center the data
    X_centered = X - mean  # Shape: (num_samples, num_sims, dim)
    
    # Compute covariance matrix per simulation
    # cov = (X^T @ X) / (num_samples - 1) for each simulation
    cov_matrices = torch.einsum('nsd, nse -> sde', X_centered, X_centered) / (num_samples - 1)
    
    return cov_matrices  # Shape: (num_sims, dim, dim)

def compute_fom(samples):
    """
    Compute the Figure of Merit (FoM) given posterior samples.

    Parameters:
    samples (numpy.ndarray): An (N, d) array where N is the number of samples
                             and d is the number of parameters.

    Returns:
    float: The computed Figure of Merit (FoM).
    """
    cov_matrix = compute_cov_matrix_per_sim(samples) # Compute covariance matrix
    det_cov = torch.linalg.det(cov_matrix)  # Determinant of covariance matrix

    fom = 1.0 / (det_cov)**(samples.shape[1])  # Compute Figure of Merit 
    return fom.mean().item()