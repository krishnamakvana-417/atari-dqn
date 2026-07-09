"""
All hyperparameters live here. Change these instead of digging through
train.py / agent.py when you want to tune something.
"""

# ---- Environment ----
ENV_ID = "ALE/Pong-v5"
FRAME_STACK = 4          # how many frames stacked together as one state
FRAME_SKIP = 4            # repeat each action for N frames (standard DQN preprocessing)
SCREEN_SIZE = 84          # resize frames to SCREEN_SIZE x SCREEN_SIZE

# ---- Replay buffer ----
BUFFER_SIZE = 100_000      # number of transitions to store (reduce if you run out of RAM)
BATCH_SIZE = 32            # minibatch size sampled from buffer for each learning step
MIN_BUFFER_SIZE = 10_000   # don't start learning until buffer has at least this many transitions

# ---- Training ----
TOTAL_STEPS = 1_000_000    # total environment steps to train for
LEARNING_RATE = 1e-4
GAMMA = 0.99                # discount factor
TARGET_UPDATE_FREQ = 10_000  # sync target network every N steps
TRAIN_FREQ = 4               # call agent.learn() every N environment steps

# ---- Exploration (epsilon-greedy) ----
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 1_000_000  # linearly decay epsilon over this many steps

# ---- Logging / checkpoints ----
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
CHECKPOINT_FREQ = 50_000    # save a checkpoint every N steps
LOG_FREQ = 10               # print/log progress every N episodes

# ---- Device ----
DEVICE = "cuda"  # will fall back to "cpu" automatically in code if cuda isn't available
