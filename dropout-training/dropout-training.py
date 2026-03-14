import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng is None:
        mask = np.random.random(size=x.shape)
    else:
        mask = rng.random(size=x.shape)
    
    mask = mask < (1 - p)
    mask = mask / (1 - p)

    output = x * mask

    return (output, mask)