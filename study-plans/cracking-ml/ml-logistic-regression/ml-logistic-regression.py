import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape 
    w = np.zeros(d, dtype=float)
    b = 0.0

    for _ in range(n_iters):
        z = X @ w + b
        y_hat = 1 / (1 + np.exp(-z))
        loss = y_hat - y
        dw = (X.T @ loss) / n
        db = np.mean(loss)
        w -= lr * dw
        b -= lr * db

    return (w, b)
