from scipy.stats.qmc import Sobol
import numpy as np

def draw_sobol_samples(prior: dict, num_samples: int) -> np.ndarray:
    """
    Draw Sobol samples from the prior distribution.
    
    Args:
        prior: The prior distribution object.
        
    Returns:
        A numpy array of Sobol samples.
    """
    # Create a Sobol sequence generator
    nd = len(prior)
    sobol_samples = Sobol(d=nd, scramble=True, seed=100)
    
    # Draw samples from the Sobol sequence
    samples = sobol_samples.random(n=num_samples)
    
    widths = np.array([prior[key][1] - prior[key][0] for key in prior])
    lower = np.array([prior[key][0] for key in prior])
    # Transform the samples according to the prior distribution
    transformed_samples = samples*widths + lower
    
    return transformed_samples