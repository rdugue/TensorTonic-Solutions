def expected_sarsa_update(Q, transitions, policy, alpha, gamma):
    """
    Returns: 2D list of shape (S, A), updated Q values rounded to 4 decimals
    """
    for s, a, r, sp in transitions:
        total = sum(p * q for p, q in zip(policy[sp], Q[sp]))
        target = gamma * total
        error = r + target - Q[s][a]
        Q[s][a] += alpha * error
    return [[round(float(q), 4) for q in r] for r in Q]