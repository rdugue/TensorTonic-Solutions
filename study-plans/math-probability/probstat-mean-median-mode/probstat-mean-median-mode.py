import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    mode = Counter(x).most_common(1)[0][0]
    x = np.asarray(x, dtype=float)

    return {
        'mean': np.mean(x),
        'median': np.median(x),
        'mode': mode
    }