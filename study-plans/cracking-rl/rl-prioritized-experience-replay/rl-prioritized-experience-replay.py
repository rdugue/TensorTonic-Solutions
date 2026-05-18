def per_priorities_and_weights(td_errors, alpha, beta, epsilon=1e-6):
    """
    Returns: tuple (probs, is_weights), both lists of length N rounded to 4 decimals
    """
    N = len(td_errors)
    probs = [(abs(t) + epsilon) ** alpha for t in td_errors]
    p_sum = sum(probs)
    probs = [p / p_sum for p in probs]
    weights = [(N * p) ** -beta for p in probs]
    w_max = max(weights)
    weights = [w / w_max for w in weights]
    return probs, weights