import numpy as np

def mahalanobis_distance(x, mean, cov):
    """
    Returns: float, the Mahalanobis distance from x to the distribution.
    """
    x = np.array(x, dtype=np.float64)
    mean = np.array(mean, dtype=np.float64)
    cov = np.array(cov, dtype=np.float64)

    diff = x - mean
    cov_inv = np.linalg.inv(cov)

    return np.sqrt(diff @ cov_inv @diff.T)