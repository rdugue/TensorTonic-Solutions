import math 

def ucb1_scores(Q, N, t, c):
    """
    Returns: list of K UCB1 scores, each rounded to 4 decimals
    """
    K = len(Q)
    UCB1 = [0] * K

    for a in range(K):
        ratio = math.log(t) / N[a]
        UCB1[a] = round(Q[a] + c * math.sqrt(ratio),4)

    return UCB1