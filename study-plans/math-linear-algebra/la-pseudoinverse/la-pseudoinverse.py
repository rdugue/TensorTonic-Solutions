import numpy as np

def pseudoinverse(A):
    """
    Returns: ndarray, the Moore-Penrose pseudoinverse of A.
    """
    A = np.array(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.zeros_like(s)
    idx = s > 1e-10
    s_inv[idx] = 1.0 / s[idx]
    sp = np.diag(s_inv)
    return Vt.T @ sp @ U.T