import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    y = np.asarray(y, dtype=np.float64)
    W = np.zeros(n, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        for i in range(m):
            z = np.dot(W, X[i]) + b
            y_hat = 1.0 if z >= 0 else 0.0
            error = y[i] - y_hat
            W += lr * error * X[i]
            b += lr * error

    return (W.tolist(), b)
         
    