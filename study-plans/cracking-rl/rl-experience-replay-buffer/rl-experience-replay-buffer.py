def replay_buffer_sample(capacity, transitions, sample_indices):
    """
    Returns: tuple (states, actions, rewards, next_states, dones), each list of length len(sample_indices)
    """
    buffer = [[] for _ in range(capacity)]
    out = ([], [], [], [], [])
    head = 0

    for t in transitions:
        buffer[head] = t
        head = (head + 1) % capacity

    for idx in sample_indices:
        sample = buffer[idx]
        for i, x in enumerate(sample):
            out[i].append(x if i != 2 else float(x))

    return out