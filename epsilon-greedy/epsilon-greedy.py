import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    rng = rng if rng is not None else np.random.default_rng()
    prob = rng.random()

    if prob >= epsilon:
        a = np.argmax(q_values)
    else:
        a = rng.integers(low=0, high=len(q_values)) 

    return a