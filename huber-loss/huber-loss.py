import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    e = y_true - y_pred
    abs_e = np.abs(e)
    L2 = e ** 2 / 2
    L1 = delta * (abs_e - (delta / 2))
    samples = np.where(abs_e > delta, L1, L2)
    return np.mean(samples)