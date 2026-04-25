import numpy as np

def cholesky_decompose(A):
    """
    Returns: lower triangular L where A = L @ L.T, or None if not positive definite.
    """
    A = np.array(A, dtype=np.float64)
    m, n = A.shape 
    L = np.zeros((m, n))

    for j in range(n):
        L[j, j] = A[j, j] - np.sum(L[j, :j] ** 2)
        if L[j, j] <= 0:
            return None
        L[j, j] = np.sqrt(L[j, j])
        for i in range(m):
            if i > j:       
                L[i, j] = (A[i, j] - np.sum(L[i, :j] @ L[j, :j])) / L[j, j]

    return L