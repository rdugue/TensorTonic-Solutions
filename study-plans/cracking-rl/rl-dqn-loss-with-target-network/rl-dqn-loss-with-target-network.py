def dqn_loss(Q_online, Q_target_next, actions, rewards, dones, gamma):
    """
    Returns: float, mean squared TD error rounded to 4 decimals
    """
    B = len(rewards)
    losses = []

    for i in range(B):
        r, d = rewards[i], dones[i]
        target = max(Q_target_next[i])
        q_value = Q_online[i][actions[i]]
        loss = r + gamma * (1 - d) * target - q_value
        losses.append(loss ** 2)

    L = sum(losses) / B
    return round(L, 4)