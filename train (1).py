"""
train.py
Initializes the LunarLander-v3 environment and trains a PPO agent
using Stable-Baselines3. Training metrics are logged to a CSV via
the Monitor wrapper, and the final policy is saved to models/.
"""

import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

TOTAL_TIMESTEPS = 500_000
MODEL_DIR = "models"
LOG_DIR = "logs"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_lunarlander")
LOG_PATH = os.path.join(LOG_DIR, "training_monitor.csv")

# PPO hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 64
N_STEPS = 2048
N_EPOCHS = 10
CLIP_RANGE = 0.2


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Create and wrap the environment
    env = gym.make("LunarLander-v3")
    env = Monitor(env, filename=LOG_PATH)

    # 2. Initialize the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS,
        clip_range=CLIP_RANGE,
        verbose=1,
    )

    # 3. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 4. Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")
    print(f"Training log saved to {LOG_PATH}")

    env.close()


if __name__ == "__main__":
    main()
