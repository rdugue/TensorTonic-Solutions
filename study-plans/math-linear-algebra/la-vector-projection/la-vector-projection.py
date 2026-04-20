import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    result = (u @ v) / (v @ v) 
    return result * v