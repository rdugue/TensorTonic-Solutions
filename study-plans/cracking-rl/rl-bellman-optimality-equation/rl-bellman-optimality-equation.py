def bellman_optimality_backup(P, R, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    V_new = []

    for s in range(S):
        A = len(P[s])
        values = []
        
        for a in range(A):
            total = 0.0
            
            for sp in range(S):
                probs = P[s][a][sp]
                reward = R[s][a][sp] + gamma * V[sp]
                total += probs * reward
                
            values += [total]
            
        V_new += [max(values)]

    return V_new