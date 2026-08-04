import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    variance = np.sum((x - np.mean(x)) ** 2) / (n - 1)
    dev = np.sqrt(variance)

    return {
        'variance': variance,
        'std_dev': dev
    }