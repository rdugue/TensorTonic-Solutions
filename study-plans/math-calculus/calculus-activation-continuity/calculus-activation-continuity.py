import numpy as np

def activation_continuity_analysis(x):
    """
    Returns: dict mapping 'relu', 'leaky_relu', 'gelu' to lists of non-differentiable x values
    """
    x = np.array(x, dtype=np.float64)
    
    relu = x[x == 0.0]

    leaky = x[x == 0.0]

    gelu = np.array([])

    return {
        'relu': relu,
        'leaky_relu': leaky,
        'gelu': gelu
    }