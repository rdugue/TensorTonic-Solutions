def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    num_t = len(rewards)
    A = [0] * num_t
    last_adv = 0

    for t in reversed(range(num_t)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        A[t] = delta + gamma * lam * last_adv
        last_adv = A[t]

    return A