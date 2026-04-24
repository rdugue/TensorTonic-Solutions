import numpy as np

def qr_decompose(A):
    """
    Returns: tuple (Q, R) where A = Q @ R.
    """
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        if j > 0:
            R[:j, j] = Q[:, :j].T @ v
            v -= Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R