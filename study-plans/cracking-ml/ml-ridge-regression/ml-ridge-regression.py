def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = X.shape
    W = np.zeros(n, dtype=float)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ W + b
        error = y_hat - y
        dw = 2 / m * (X.T @ error) + 2 * alpha * W
        db = 2 / m * np.sum(error)
        W -= lr * dw
        b -= lr * db

    return np.round(W, 4), round(b, 4)