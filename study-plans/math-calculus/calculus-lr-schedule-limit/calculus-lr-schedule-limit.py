import numpy as np

def lr_schedule_analysis(alpha_0, k):
    """
    Returns: dict with 'limit' (float), 'sum_diverges' (bool), 'sum_sq_converges' (bool)
    """
    limit = 0.0 if alpha_0 > 0 and k > 0 else float(alpha_0)
    diverges = alpha_0 > 0
    converges = alpha_0 == 0 or k > 0
    return {'limit': limit, 'sum_diverges': diverges, 'sum_sq_converges': converges}