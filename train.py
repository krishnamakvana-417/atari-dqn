"""
Main training entrypoint. Run with: python train.py

This file is orchestration only — env setup, main loop, logging,
checkpointing. All the real DQN logic lives in agent.py.
"""

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import config
from agent import DQNAgent


def make_env(render_mode=None):
    """
    Builds the preprocessed Atari environment:
    - AtariPreprocessing handles: grayscale, resize to 84x84, frame-skip,
      max-pooling over skipped frames (standard DQN preprocessing).
    - FrameStackObservation stacks the last N frames into one observation.
    """
    env = gym.make(config.ENV_ID, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=config.FRAME_SKIP,
        screen_size=config.SCREEN_SIZE,
        grayscale_obs=True,
        scale_obs=False,  # keep as uint8, agent normalizes to [0,1] itself
    )
    env = FrameStackObservation(env, stack_size=config.FRAME_STACK)
    return env


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    env = make_env(render_mode=None)  # headless for training speed
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape  # (4, 84, 84)

    print(f"Environment: {config.ENV_ID}")
    print(f"Actions: {num_actions}, State shape: {state_shape}")

    agent = DQNAgent(num_actions, state_shape, config)
    print(f"Using device: {agent.device}")

    episode_rewards = []
    episode_reward = 0.0
    episode_count = 0
    losses = []

    state, _ = env.reset()
    start_time = time.time()

    for step in range(1, config.TOTAL_STEPS + 1):
        agent.steps_done = step

        action = agent.select_action(np.array(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        episode_reward += reward

        if step % config.TRAIN_FREQ == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        if step % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)

            if episode_count % config.LOG_FREQ == 0:
                avg_reward = np.mean(episode_rewards[-config.LOG_FREQ:])
                avg_loss = np.mean(losses[-1000:]) if losses else 0.0
                elapsed = time.time() - start_time
                print(
                    f"Episode {episode_count} | Step {step}/{config.TOTAL_STEPS} | "
                    f"Avg Reward (last {config.LOG_FREQ}): {avg_reward:.2f} | "
                    f"Epsilon: {agent.epsilon():.3f} | Avg Loss: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed/60:.1f} min"
                )

            episode_reward = 0.0
            state, _ = env.reset()

        if step % config.CHECKPOINT_FREQ == 0:
            path = os.path.join(config.CHECKPOINT_DIR, f"dqn_step{step}.pt")
            agent.save(path)
            print(f"Saved checkpoint: {path}")

    # final save
    final_path = os.path.join(config.CHECKPOINT_DIR, "dqn_final.pt")
    agent.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
