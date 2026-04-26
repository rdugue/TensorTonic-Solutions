import numpy as np

def pca_project(X, n_components):
    """
    Returns: ndarray of shape (n_samples, n_components), the projected data.
    """
    X = np.array(X, dtype=np.float64)
    n = X.shape[0]
    centered = X - np.mean(X, axis=0)
    C = (centered.T @ centered) / (n - 1)
    vals, V = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    V = V[:, idx]
    return centered @ V[:, :n_components]