import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    a = [A[i][i] for i in range(len(A))]
    return np.sum(a)

    