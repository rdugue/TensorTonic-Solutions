import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = X.shape
    W = np.zeros(n)
    b = 0.0

    for _ in range(n_iters):
        z = X @ W + b
        y_hat = 1 / (1 + np.exp(-z))
        error = y_hat - y
        dw = (X.T @ error) / m
        db = np.mean(error)
        W -= lr * dw
        b -= lr * db

    weights = np.round(W, 4)
    bias = round(b, 4)
    return weights, bias
