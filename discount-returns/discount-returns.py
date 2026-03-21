def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    G = [0] * len(rewards)
    current = 0

    for t in reversed(range(len(rewards))):
        G[t] = rewards[t] + gamma * current
        current = G[t]

    return G