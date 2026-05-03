import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    N = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = y_pred[np.arange(N), y_true]
    return np.mean(-np.log(correct + 1e-15))