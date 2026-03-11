import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    m = len(A)
    n = len(A[0])
    result = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            result[i][j] = A[j][i]

    return result