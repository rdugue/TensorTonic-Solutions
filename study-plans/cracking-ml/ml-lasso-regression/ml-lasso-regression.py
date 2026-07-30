def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = X.shape
    W = np.zeros(n, dtype=float)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ W + b
        error = y_hat - y
        dw = 2 / m * (X.T @ error) + alpha * np.sign(W)
        db = 2 / m * np.sum(error)
        W -= lr * dw
        b -= lr * db

    return np.round(W, 4), round(b, 4)