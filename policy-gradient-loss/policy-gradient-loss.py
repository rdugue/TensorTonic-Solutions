def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    num_t = len(rewards)
    G = [0] * num_t

    current = 0
    for t in reversed(range(num_t)):
        current = rewards[t] + gamma * current
        G[t] = current
        
    g_m = sum(G) / num_t

    A = [g - g_m for g in G]
    probs_a = [p * a for p, a in zip(log_probs, A)]

    L = -sum(probs_a) / num_t

    return L