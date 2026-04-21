import numpy as np

def lu_decomposition(A):
    """
    Returns: tuple (L, U) where A = L @ U.
    """
    A = np.array(A, dtype=np.float64)
    n = len(A)
    L = np.eye(n)
    U = np.triu(A)

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - np.sum(L[k, :k] * U[:k, j])
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.sum(L[i, :k] * U[:k, k])) / U[k, k]

    return L, U