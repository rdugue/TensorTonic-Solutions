def sarsa_update(Q, transitions, alpha, gamma):
    """
    Returns: 2D list of shape (S, A), updated Q values rounded to 4 decimals
    """
    for s, a, r, sp, ap in transitions:
        error = float(r) + float(gamma) * Q[sp][ap] - Q[s][a]
        Q[s][a] += alpha * error
    return [[round(float(q), 4) for q in r] for r in Q]