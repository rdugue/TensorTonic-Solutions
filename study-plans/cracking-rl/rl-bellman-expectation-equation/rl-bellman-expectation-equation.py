def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    V_new = [0.0] * S

    for s in range(S):
        actions = policy[s]
        total = 0.0
        
        for a in range(len(actions)):
            a_total = 0.0
            
            for sp in range(S):
                ps = P[s][a][sp]
                r = float(R[s][a][sp])
                old = float(V[sp])
                a_total += ps * (r + gamma * old)
                
            total += actions[a] * a_total
            
        V_new[s] = round(total, 4)

    return V_new