import numpy as np

def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Returns: list of floats
    """
    rng = np.random.default_rng(42)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    losses = []

    for _ in range(epochs):
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            activations, denses = forward(x_batch, weights, biases)
            y_preds = activations[-1]
            dL_dZ = (y_preds - y_batch) / len(x_batch)
            backward(dL_dZ, activations, denses, weights, biases, lr)
            
        activations, _ = forward(X, weights, biases)
        y_preds = activations[-1]
        L = np.sum((y_preds  - y) ** 2) / (2 * len(X))
        losses.append(round(float(L), 4))
        
    return losses

def forward(X, weights, biases):
    layers = len(biases)
    pre = []
    a = X
    activations = [a]

    for l in range(layers):
        w = np.array(weights[l], dtype=float)
        b = np.array(biases[l], dtype=float)
        z = np.dot(a, w.T) + b
        pre.append(z) 
        a = np.maximum(0, z) if l < layers - 1 else z
        activations.append(a)

    return activations, pre
    
def backward(dL_dZ, a, d, weights, biases, lr):
    layers = len(biases)
    
    for l in reversed(range(layers)):
        a_prev = a[l]
        W = weights[l]
        b = biases[l]
        dW = dL_dZ.T @ a_prev
        db = np.sum(dL_dZ, axis=0)
        weights[l] = (W -lr * dW).tolist()
        biases[l] = ( b -lr * db).tolist()

        if l > 0:
            dL_dZ = dL_dZ @ W
            z_prev = np.array(d[l-1], dtype=float)
            dL_dZ *= (z_prev > 0) 