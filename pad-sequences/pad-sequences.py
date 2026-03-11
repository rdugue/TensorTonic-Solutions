import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if max_len is not None else max(len(seq) for seq in seqs)

    result = np.full((N, L), pad_value)

    for i in range(N):
        r = min(L, len(seqs[i]))
        for j in range(r):
            result[i][j] = seqs[i][j]
        
    return result