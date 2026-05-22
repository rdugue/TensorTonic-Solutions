def dueling_q_values(V, A_advantages):
    """
    Returns: 2D list of shape (B, num_actions), Q values rounded to 4 decimals
    """
    S = len(V)
    A = len(A_advantages[0])
    Q = []
    for s in range(S):
        mean = sum(A_advantages[s]) / A
        Q.append([])
        for a in range(A):
            q_sa = V[s] + A_advantages[s][a] - mean
            Q[s].append(round(q_sa, 4))
    return Q