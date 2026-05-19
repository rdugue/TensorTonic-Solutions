def double_dqn_targets(Q_online_next, Q_target_next, rewards, dones, gamma):
    """
    Returns: list of length B, Double DQN targets rounded to 4 decimals
    """
    B = len(rewards)
    out = [0.0] * B
    for i in range(B):
        r, d = rewards[i], dones[i]
        a = Q_online_next[i].index(max(Q_online_next[i]))
        out[i] = round(r + gamma * Q_target_next[i][a] * (1 - d), 4)
    return out
        
