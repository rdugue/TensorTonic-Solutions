import numpy as np
from collections import Counter

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """
    x = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    xt = np.asarray(X_test, dtype=float)
    preds = []

    for t in xt:
        N = np.sqrt(np.sum((x - t) ** 2, axis=1))
        idx = np.argsort(N)[:k]
        labels = y[idx].tolist()
        counter = Counter(labels)
        most = max(counter.values())
        best = min(label for label, c in counter.items() if c == most)
        preds.append(best)

    return preds