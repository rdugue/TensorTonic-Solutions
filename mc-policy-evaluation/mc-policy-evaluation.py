import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    g_sums = np.zeros(n_states)
    g_counts = np.zeros(n_states)

    for episode in episodes:
        G = 0
        S = [step[0] for step in episode]

        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma * G
            if state not in S[:t]:
                g_sums[state] += G
                g_counts[state] += 1

    V = np.divide(g_sums, g_counts, out=np.zeros_like(g_sums), where=g_counts!=0)
    return V
                
            
