import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = X.shape
    W = np.zeros(n, dtype=float)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ W + b
        error = y_hat - y
        dw = 2 / m * (X.T @ error)
        db = 2 / m * np.sum(error)
        W -= lr * dw
        b -= lr * db

    weights = np.round(W, 4)
    bias = round(b, 4)
    return weights, bias
