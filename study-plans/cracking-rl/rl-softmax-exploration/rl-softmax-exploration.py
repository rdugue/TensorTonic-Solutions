from math import exp
def softmax_action_probs(Q_values, tau):
    """
    Returns: list of length A, action probabilities under softmax/Boltzmann, rounded to 4 decimals
    """
    A = len(Q_values)
    policy = [0.0] * A
    
    for a in range(A):
        xp = [exp((qa - max(Q_values)) / tau) for qa in Q_values]
        policy[a] = round(xp[a] / sum(xp), 4)

    return policy