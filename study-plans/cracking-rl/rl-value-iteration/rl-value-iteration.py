def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])
    V = [0.0] * S
    policy = [0] * S

    for _ in range(max_iters):
        V_ = [0.0] * S
        for s in range(S):
            values = []
            for a in range(A):
                q_sa = 0.0
                for sp in range(S):
                    prob = P[s][a][sp]
                    reward = R[s][a][sp] + gamma * V[sp]
                    q_sa += prob * reward
                values += [q_sa]
            V_[s] = max(values)
        delta = max(abs(v_ - v) for v_, v in zip(V_, V))
        V = V_
        if delta < tol:
            break
    
    for s in range(S):
        values = []             
        for a in range(A):
            q_sa = 0.0
            for sp in range(S):
                prob = P[s][a][sp]
                reward = R[s][a][sp] + gamma * V[sp]
                q_sa += prob * reward
            values += [q_sa]
        policy[s] = values.index(max(values))

    return [round(float(val), 4) for val in V], policy