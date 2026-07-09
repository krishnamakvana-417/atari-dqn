"""
Load a checkpoint and watch the agent play in a live window.
Run with: python evaluate.py --checkpoint checkpoints/dqn_final.pt

This is also useful early on with a fresh/random agent (no --checkpoint)
just to sanity check that the environment and action space work as expected.
"""

import argparse
import time
import numpy as np

import config
from agent import DQNAgent
from train import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a saved checkpoint (.pt). If omitted, uses a random/untrained agent.")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_env(render_mode="human")  # pops up a live window
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape

    agent = DQNAgent(num_actions, state_shape, config)
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint given — watching an untrained (random) agent.")

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(np.array(state), greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            time.sleep(0.01)  # slight delay so it's watchable, not a blur

        print(f"Episode {ep}: reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()
