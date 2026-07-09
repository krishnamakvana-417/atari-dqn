# Atari DQN — Pong

A from-scratch implementation of Deep Q-Network (Mnih et al., 2015) trained to play Atari Pong,
built as part of Summer of Code.

## Setup

```bash
pip install -r requirements.txt
```

Note: `gymnasium[atari]` requires accepting the Atari ROM license. If prompted, run:
```bash
pip install gymnasium[accept-rom-license]
```

## Project structure

- `config.py` — all hyperparameters (learning rate, epsilon schedule, buffer size, etc.)
- `network.py` — the CNN Q-network architecture
- `replay_buffer.py` — experience replay memory
- `agent.py` — DQN agent: action selection, learning step, target network sync
- `train.py` — main training loop entrypoint
- `evaluate.py` — load a checkpoint and watch the agent play in a live window

## Training

```bash
python train.py
```

Runs headless (no window) for speed. Progress is printed every `LOG_FREQ` episodes,
checkpoints saved to `checkpoints/` every `CHECKPOINT_FREQ` steps.

Training Pong to a reasonably strong policy typically takes ~1-2M environment steps,
which is a few hours on GPU and considerably longer on CPU.

## Watching the agent play

```bash
python evaluate.py --checkpoint checkpoints/dqn_final.pt
```

Opens a live pygame window. Omit `--checkpoint` to watch an untrained (random) agent —
useful as a first sanity check that the environment works.

## Results

_(fill in once trained: reward curve, before/after episode videos or screenshots, final avg score)_

## References

- Mnih et al., 2015 — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
