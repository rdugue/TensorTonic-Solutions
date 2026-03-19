import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    T = len(episodes)
    V = np.zeros(n_states)
    R = np.zeros(n_states)
    count = np.zeros(n_states)

    for episode in episodes:
        S = [step[0] for step in episode]
        G = 0
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward
            if state not in S[:t]:
                R[state] += G
                count[state] += 1
                
    for state in range(n_states):
        if count[state] > 0:
            V[state] = R[state] / count[state]
        
    return V