def mc_policy_evaluation(episodes, gamma, num_states):
    """
    Returns: list of length num_states, V[s] rounded to 4 decimals; unvisited states are 0.0
    """
    V = [0.0] * num_states
    G_t = [0.0] * num_states
    counts = [0] * num_states
    
    for episode in episodes:
        T = len(episode)
        episode_states = [step[0] for step in episode]
        G = 0.0
        for t in reversed(range(T)):
            state, reward = episode[t]
            G = reward + gamma * G
            if state not in episode_states[:t]:
                G_t[state] += G
                counts[state] += 1

    for s in range(num_states):
        if counts[s] > 0:
            V[s] = G_t[s] / counts[s]
        
    return V