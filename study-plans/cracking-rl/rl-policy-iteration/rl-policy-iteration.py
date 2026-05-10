def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])
    V = [0.0] * S
    policy = [0] * S

    for _ in range(max_iters):
        while True:
            V_new = [0.0] * S
            for s in range(S):
                a = policy[s]
                for sp in range(S):
                    q_sa = P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
                    V_new[s] += q_sa
            delta = max(abs(a - b) for a, b in zip(V_new, V))
            V = V_new
            if delta < eval_tol:
                break

        while True:
            stable = True
            for s in range(S):
                q_values = []
                for a in range(A):
                    q_sa = 0.0
                    for sp in range(S):
                        q_sa += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
                    q_values.append(q_sa)
                old_action = policy[s]
                policy[s] = q_values.index(max(q_values))
                stable = old_action == policy[s]
            if stable:
               break 
                
    return V, policy