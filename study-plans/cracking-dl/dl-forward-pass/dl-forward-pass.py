import numpy as np

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=float)

    layers = len(biases)
    activations = []
    pre_activations = []

    for l in range(layers):
        W = np.array(weights[l], dtype=float)
        b = np.array(biases[l], dtype=float)
        copy = np.round(x.copy(), 4).tolist()
        activations.append(copy)
        x = np.dot(W, x) + b
        copy = np.round(x.copy(), 4).tolist()
        pre_activations.append(copy)
        if l < layers-1:
            x = np.maximum(0, x)
    copy = np.round(x.copy(), 4).tolist()
    activations.append(copy)

    return {
        'activations': activations,
        'pre_activations': pre_activations
    }