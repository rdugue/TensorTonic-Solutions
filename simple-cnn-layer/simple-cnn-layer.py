import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    N, _, H, W_in = x.shape

    C_out, _, KH, KW = W.shape

    H_out = H - KH + 1
    W_out = W_in - KW + 1

    output = np.zeros((N, C_out, H_out, W_out), float)

    for out in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                cell = np.sum(x[:, :, i:KH+i, j:KW+j] * W[out], axis=(1, 2, 3))
                cell += b[out]
                output[:, out, i, j] = cell

    return output