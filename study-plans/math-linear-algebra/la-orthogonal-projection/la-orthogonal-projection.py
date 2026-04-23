import numpy as np

def projection_matrix(A):
    """
    Returns: ndarray, the projection matrix onto the column space of A.
    """
    A = np.array(A, dtype=np.float64)
    return A @ np.linalg.pinv(A.T @ A) @ A.T