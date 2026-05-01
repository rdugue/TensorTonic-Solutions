import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = np.asarray(x, dtype=np.float64)
    output = None
    derivative = None
    if activation == 'relu':
        output = np.maximum(0, x)
        derivative = (x > 0).astype(float)
    elif activation == 'leaky_relu':
        output = np.where(x > 0, x, 0.01 * x)
        derivative = np.where(x > 0, 1, 0.01)
    elif activation == 'sigmoid':
        output = 1 / (1 + np.exp(-x))
        derivative = output * (1 - output)
    elif activation == 'tanh':
        output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 
        derivative = 1 - output ** 2
    elif activation == 'gelu':
        tahn_x = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        Z = (np.exp(tahn_x) - np.exp(-tahn_x)) / (np.exp(tahn_x) + np.exp(-tahn_x))
        output = 0.5 * x * (1 + Z)
        derivative = 0.5 * (1 + Z) + 0.5 * x * (1 - Z ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    else:
        Z = 1 / (1 + np.exp(-x))
        output = x * Z
        derivative = Z + x * Z * (1 - Z)

    return [output, derivative]
        