import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X, dtype=np.float64)
    m, n = X.shape
    y = np.array(y, dtype=np.float64)
    W = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ W + b
        error = y_hat - y
        dl_dw = (X.T @ error) * 2 / m
        dl_db = np.sum(error)  * 2 / m
        W -= lr * dl_dw
        b -= lr * dl_db

    return (W.tolist(), b)