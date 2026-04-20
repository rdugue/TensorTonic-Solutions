import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    V = np.array(vectors, dtype=np.float64)
    Q = np.zeros_like(V)

    for i in range(len(V)):
        n = 0
        for j in range(i):
            n += (V[i] @ V[j]) / (V[j] @ V[j]) * V[j]
            
        V[i] -= n
        Q[i] =  V[i] / np.linalg.norm(V[i])

    return Q