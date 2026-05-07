def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    T = len(rewards)
    G_t = [0.0] * T
    
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        G_t[t] = round(G, 4)

    return G_t