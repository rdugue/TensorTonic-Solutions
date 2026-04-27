import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: ndarray, the attention output softmax(Q @ K.T / sqrt(d_k)) @ V.
    """
    Q = np.array(Q, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    dk = Q.shape[1]

    S = (Q @ K.T) / np.sqrt(dk)
    num = np.exp(S - np.max(S))
    axis = None if S.ndim < 2 else 1
    denom = np.sum(num, axis=axis, keepdims=True)

    return (num / denom) @ V