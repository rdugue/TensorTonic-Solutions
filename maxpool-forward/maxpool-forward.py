def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    H, W = len(X), len(X[-1])
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = []

    for i in range(H_out):
        out.append([])
        for j in range(W_out):
            max_cell = float("-inf")
            for a in range(pool_size):
                for b in range(pool_size):
                    current = X[i * stride + a][j * stride + b]
                    max_cell = max(max_cell, current)
            out[i].append(max_cell)
            
    return out 
    