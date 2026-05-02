import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=np.float64)
    mask = np.array(mask, dtype=np.float64)
    if mode == 'train':
        X = (X * mask) / (1 - drop_prob)
    return X.tolist()