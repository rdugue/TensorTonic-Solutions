import numpy as np

def skewness_kurtosis(data):
    """
    Returns: dict with 'skewness', 'kurtosis', and interpretation strings.
    """
    x = np.asarray(data, dtype=float)
    n = len(data)
    mean = np.mean(x)
    s = np.std(x, ddof=1)

    g1 = n / ((n - 1) * (n - 2)) * np.sum(((x - mean) / s) ** 3)
    
    a = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    b = np.sum(((x - mean) / s) ** 4)
    c = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    g2 = a * b - c

    g1_int = "approximately symmetric"
    if g1 > 0.5:
        g1_int = "right-skewed"
    if g1 < -0.5:
        g1_int = "left-skewed"
    
    g2_int = "mesokurtic"
    if g2 > 1:
        g2_int = "leptokurtic"
    if g2 < -1:
        g2_int = "platykurtic"

    return {
        'skewness': round(g1, 4),
        'kurtosis': round(g2, 4),
        'skew_interpretation': g1_int,
        'kurtosis_interpretation': g2_int
    }
    