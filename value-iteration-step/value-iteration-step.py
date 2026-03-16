def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    S = len(values)
    n_values = [0.0] * S
    
    for s in range(S):
        A = len(rewards[s])
        Q = []
        for a in range(A):
            S_n = len(transitions[s][a])
            t_v = 0
            for s_n in range(S_n):
                t_v += transitions[s][a][s_n] * values[s_n]
            Q.append(rewards[s][a] + gamma * t_v)
        n_values[s] = max(Q)
            
    return n_values