def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    n = len(priorities)
    
    powers = [p**alpha for p in priorities]
    P = [p / sum(powers) for p in powers]

    weights = [(n * p)**-beta for p in P]
    W = [w / max(weights) for w in weights]

    return [P, W]