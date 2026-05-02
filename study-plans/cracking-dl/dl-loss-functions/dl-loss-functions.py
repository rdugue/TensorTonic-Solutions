import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y = np.array(y_true, dtype=float)
    y_hat = np.array(y_pred, dtype=float)
    clip = 1e-15
    if loss_type == 'mse':
        return np.mean((y - y_hat) ** 2)
    if loss_type == 'bce':
        y_hat = np.clip(y_hat, clip, 1 - clip)
        return -np.mean(y * np.log(y_hat) + (1 - y + clip) * np.log(1 - y_hat))
    if loss_type == 'cce':     
        max_a = np.max(y_hat, axis=1, keepdims=True)
        softmax = np.exp(y_hat - max_a) / np.sum(np.exp(y_hat - max_a), axis=1, keepdims=True)
        correct = softmax[np.arange(len(y)), y.astype(int)]
        return -np.mean(np.log(correct + clip))
    return np.mean(np.maximum(0, 1 - y * y_hat))
        
    