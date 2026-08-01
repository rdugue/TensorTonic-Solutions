import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    W = np.zeros((n, n_classes), dtype=float)
    b = np.zeros(n_classes, dtype=float)
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y] = 1.0

    for _ in range(n_iters):
        z = X @ W + b
        norm = z - np.max(z, axis=1, keepdims=True)
        P = np.exp(norm) / np.sum(np.exp(norm), axis=1, keepdims=True)
        error = P - y_one_hot
        dw = (X.T @ error) / m
        db = np.mean(error, axis=0)
        W -= lr * dw
        b -= lr * db

    return W, b