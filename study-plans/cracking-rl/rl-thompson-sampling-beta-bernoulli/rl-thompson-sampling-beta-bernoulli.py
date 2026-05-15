def thompson_posterior(K, observations, alpha_prior=1.0, beta_prior=1.0):
    """
    Returns: tuple (alpha_list, beta_list), each a list of length K rounded to 4 decimals
    """
    alpha = [float(alpha_prior)] * K
    beta = [float(beta_prior)] * K

    for arm, r in observations:
        alpha[arm] += r
        beta[arm] += (1 - r)

    return alpha, beta