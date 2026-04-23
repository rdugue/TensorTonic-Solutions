import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.array(A, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx =np.argsort(-np.abs(eigenvalues))
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    norms = np.linalg.norm(eigenvectors, axis=0)
    eigenvectors = np.divide(eigenvectors, norms, where=norms > 1e-12)
    return eigenvalues, eigenvectors