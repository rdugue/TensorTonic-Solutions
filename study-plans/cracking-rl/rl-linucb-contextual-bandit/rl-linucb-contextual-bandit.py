def linucb_ucb(A_list, b_list, context, alpha):
    """
    Returns: list of K LinUCB scores, each rounded to 4 decimals
    """
    K = len(A_list)
    x = np.asarray(context, dtype=float)
    scores = []

    for k in range(K):
        a = np.asarray(A_list[k], dtype=float)
        b = np.asarray(b_list[k], dtype=float)

        a_inv = np.linalg.inv(a)
        theta = a_inv @ b
        score = theta @ x + alpha * np.sqrt(x @ a_inv @ x)

        scores.append(round(score, 4))

    return scores
    