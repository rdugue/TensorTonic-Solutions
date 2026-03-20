import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    max_a = np.max(x) 

    if x.ndim < 2:
        divisor =  np.sum(np.exp(x - max_a))
    else:
        divisor =  np.sum(np.exp(x - max_a), axis=1, keepdims=True)
        
    x = np.exp(x - max_a) / divisor

    return x