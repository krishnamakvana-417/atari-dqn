"""
The Q-network: a small CNN that takes a stack of 4 grayscale frames
(4 x 84 x 84) and outputs one Q-value per possible action.

This file only defines the architecture. No training logic here —
agent.py owns two instances of this (main network + target network).
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_actions: int, input_channels: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Figure out the flattened conv output size dynamically
        # (avoids hardcoding 7*7*64 and breaking if SCREEN_SIZE changes)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out_size = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (batch, 4, 84, 84), values in [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
