def td_zero_update(V, transitions, alpha, gamma):
    """
    Returns: list of length S, updated V[s] rounded to 4 decimals
    """
    for trans in transitions:
        s, r, sp = trans
        V[s] += alpha * (r + gamma * V[sp] - V[s])

    return [round(float(v), 4) for v in V]