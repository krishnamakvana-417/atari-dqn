"""
The DQN agent: owns the main network + target network, picks actions
(epsilon-greedy), and runs the learning step (sample batch -> compute
loss -> backprop).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, num_actions: int, state_shape: tuple, config):
        self.num_actions = num_actions
        self.cfg = config

        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else "cpu"
        )

        self.q_network = QNetwork(num_actions).to(self.device)
        self.target_network = QNetwork(num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network is never trained directly

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.LEARNING_RATE
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.buffer = ReplayBuffer(config.BUFFER_SIZE, state_shape)

        self.steps_done = 0

    def epsilon(self) -> float:
        """Linearly decay epsilon from EPSILON_START to EPSILON_END."""
        cfg = self.cfg
        frac = min(1.0, self.steps_done / cfg.EPSILON_DECAY_STEPS)
        return cfg.EPSILON_START + frac * (cfg.EPSILON_END - cfg.EPSILON_START)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        state: np.uint8 array of shape (4, 84, 84)
        greedy: if True, always pick the best action (used for evaluation)
        """
        eps = 0.0 if greedy else self.epsilon()

        if random.random() < eps:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0) / 255.0
            state_t = state_t.to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """One gradient update step, sampled from the replay buffer."""
        cfg = self.cfg
        if len(self.buffer) < cfg.MIN_BUFFER_SIZE:
            return None  # not enough data yet

        states, actions, rewards, next_states, dones = self.buffer.sample(
            cfg.BATCH_SIZE
        )

        states = torch.from_numpy(states).float().to(self.device) / 255.0
        next_states = torch.from_numpy(next_states).float().to(self.device) / 255.0
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Current Q estimate for the actions actually taken
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a'), zeroed out if done
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + cfg.GAMMA * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping helps stabilize DQN training
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["q_network"])
        self.steps_done = checkpoint.get("steps_done", 0)
