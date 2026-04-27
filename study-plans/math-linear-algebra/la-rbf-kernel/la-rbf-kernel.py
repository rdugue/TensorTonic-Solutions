import numpy as np

def rbf_kernel_matrix(X, gamma):
    """
    Returns: ndarray of shape (n, n), the RBF kernel matrix.
    """
    X = np.array(X, np.float64)
    norms = np.linalg.norm(X, axis=1) ** 2
    norm_a, norm_b = norms[:, None], norms[None, :]
    D = norm_a + norm_b - 2 * X @ X.T
    D = np.maximum(D, 0)
    return np.exp(-gamma * D)