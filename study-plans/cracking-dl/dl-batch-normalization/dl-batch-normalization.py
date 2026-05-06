import numpy as np

def batch_norm(X, gamma, beta, running_mean, running_var, mode):
    """
    Returns: dict with keys "output", "running_mean", "running_var"
    """
    X = np.array(X, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    rm = np.array(running_mean, dtype=float)
    rv = np.array(running_var, dtype=float)
    eps = 1e-5
    m = 0.1

    if mode == 'train':
        mu = X.mean(axis=0)
        diff = X - mu
        var = (diff ** 2).mean(axis=0)
        x_hat = diff / np.sqrt(var + eps)
        out = gamma * x_hat + beta
        rm = (1 - m) * rm + m * mu
        rv = (1 - m) * rv + m * var
    else:
        x_hat = (X - rm) / np.sqrt(rv + eps)
        out = gamma * x_hat + beta

    return {
        "output": np.round(out, 4).tolist(), 
        "running_mean": np.round(rm, 4).tolist(), 
        "running_var": np.round(rv, 4).tolist()
    }