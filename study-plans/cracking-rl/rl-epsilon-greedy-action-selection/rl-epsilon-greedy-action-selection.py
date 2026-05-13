def epsilon_greedy_probs(Q_values, epsilon):
    """
    Returns: list of length A, action probabilities under epsilon-greedy, rounded to 4 decimals
    """
    A = len(Q_values)
    policy = [0.0] * A

    for a in range(A):
        ap = Q_values.index(max(Q_values))
        prob = epsilon / A
        policy[a] = prob if a != ap else 1 - epsilon + prob

    return policy