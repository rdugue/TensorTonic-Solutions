import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    G = np.zeros(len(rewards))
    A = np.zeros(len(rewards))

    current = 0
    for t in reversed(range(len(rewards))):
        G[t] = rewards[t] + gamma * current
        current = G[t]

    A = G - V
    return A