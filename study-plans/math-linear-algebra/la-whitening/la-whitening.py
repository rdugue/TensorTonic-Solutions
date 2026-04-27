import numpy as np

def whiten(X):
    """
    Returns: ndarray, the whitened data with identity covariance.
    """
    X = np.array(X, dtype=np.float64)
    center = X - np.mean(X, axis=0)
    C = (center.T @ center) / (X.shape[0] - 1)
    vals, V = np.linalg.eigh(C)
    vals = np.where(vals >= 1e-10, 1.0 / np.sqrt(vals), 0.0)
    D_hinv = np.diag(vals)
    return center @ V @ D_hinv