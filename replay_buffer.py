"""
Experience replay buffer. Stores (state, action, reward, next_state, done)
transitions and lets the agent sample random minibatches from them.

Random sampling breaks the correlation between consecutive frames that
you'd otherwise get from training on transitions in the order they happened.

Uses numpy arrays (not a deque of tuples) because it's much faster and more
memory-efficient once BUFFER_SIZE gets into the hundreds of thousands.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple):
        self.capacity = capacity
        self.state_shape = state_shape

        # Preallocate storage. States are stored as uint8 (0-255) to save
        # memory; convert to float and normalize when sampling.
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0          # next index to write to
        self.size = 0          # how many transitions currently stored

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
