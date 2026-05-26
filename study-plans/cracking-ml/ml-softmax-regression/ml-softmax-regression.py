import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    n, d = X.shape 
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    y_oh = np.zeros((n, n_classes))
    y_oh[np.arange(n), y] = 1.0

    for _ in range(n_iters):
        z = X @ w + b
        norm = z - np.max(z, axis=1, keepdims=True)
        y_hat = np.exp(norm) / (np.sum(np.exp(norm), axis=1, keepdims=True))
        error = y_hat - y_oh
        dw = (X.T @ error) / n
        db = np.sum(error, axis=0) / n
        w -= lr * dw
        b -= lr * db
        

    return (w, b)