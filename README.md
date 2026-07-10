# Atari Deep Q-Network (DQN) — Pong

A from-scratch implementation of the **Deep Q-Network (DQN)** algorithm proposed by Mnih et al. (2015), developed as part of the Summer of Code program. The project trains an intelligent reinforcement learning agent to play **Atari Pong** directly from raw game frames using deep learning and trial-and-error interaction.

---

# Problem Statement

Traditional supervised learning relies on labeled datasets, but game-playing agents must learn without explicit instructions. The objective of this project is to build an agent that learns how to play Atari Pong by interacting with the environment and maximizing long-term rewards.

The key challenges include:

- High-dimensional image observations
- Large state space
- Delayed rewards
- Exploration vs. exploitation
- Stable neural network training

---

# Proposed Solution

To solve this problem, the project implements the **Deep Q-Network (DQN)** algorithm.

Instead of maintaining a massive Q-table, DQN approximates the action-value function using a Convolutional Neural Network (CNN). The agent observes the game screen, predicts Q-values for every possible action, and gradually learns an optimal policy through repeated interactions with the environment.

The implementation includes several core DQN components:

- CNN-based Q-Network
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Exploration
- Temporal Difference (TD) Learning

Together, these components enable stable and efficient reinforcement learning.

---

# Project Architecture

```
Game Environment (Pong)
            │
            ▼
    Preprocessed Frames
            │
            ▼
     CNN Q-Network
            │
            ▼
 Predicted Q-values
            │
            ▼
 Epsilon-Greedy Action
            │
            ▼
 Environment Interaction
            │
            ▼
 Replay Buffer
            │
            ▼
 Network Optimization
            │
            ▼
 Periodic Target Network Update
```

---

# Project Structure

```
config.py          # Hyperparameters
network.py         # CNN architecture
replay_buffer.py   # Experience replay memory
agent.py           # DQN agent implementation
train.py           # Training pipeline
evaluate.py        # Evaluate trained agent
checkpoints/       # Saved model weights
```

---

# Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Recent versions of `ale-py` already include Atari ROMs, so no additional ROM installation or license acceptance is required.

---

# Training

Start training with:

```bash
python train.py
```

The training runs headless for faster execution.

During training:

- Episode rewards are logged periodically.
- Model checkpoints are saved automatically.
- The target network is synchronized at fixed intervals.
- Experiences are continuously stored inside the replay buffer.

Training a competitive Pong agent typically requires **1–2 million environment steps**, depending on hardware.

---

# Evaluation

Evaluate a trained model using:

```bash
python evaluate.py --checkpoint checkpoints/dqn_final.pt
```

This launches a live gameplay window where the trained agent interacts with the environment.

Running without specifying a checkpoint launches an untrained random agent for comparison.

---

# Results

The DQN agent gradually improves through experience.

Initially, the agent performs random actions and receives low rewards. As training progresses, the replay buffer accumulates diverse experiences, enabling the neural network to learn increasingly effective strategies.

Expected results include:

- Increasing average episode reward
- More consistent gameplay
- Successful ball tracking and paddle positioning
- Better decision-making compared to a random agent

*(Training reward curves, gameplay GIFs, screenshots, and final average scores can be added here after training.)*

---

# Key Features

- From-scratch DQN implementation
- CNN-based Q-value approximation
- Experience Replay for stable learning
- Target Network synchronization
- Epsilon-Greedy exploration strategy
- Modular and well-documented codebase

---

# Future Improvements

The current implementation provides a strong DQN baseline. Future enhancements may include:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Rainbow DQN
- Multi-step Learning
- Distributed training

---

# References

- Mnih et al. (2015). *Human-level Control through Deep Reinforcement Learning.*
- Gymnasium Documentation
- PyTorch Documentation
