"""
plot_training.py
Parses the Monitor CSV log produced during training and renders a
smoothed learning curve, saved to graphs/learning_curve.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_PATH = "logs/training_monitor.csv"
GRAPH_DIR = "graphs"
OUTPUT_PATH = os.path.join(GRAPH_DIR, "learning_curve.png")


def moving_average(values, window=20):
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, mode="valid")


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Monitor CSVs have a one-line JSON header comment before the actual columns
    df = pd.read_csv(LOG_PATH, skiprows=1)
    rewards = df["r"].values
    timesteps = np.cumsum(df["l"].values)

    smoothed = moving_average(rewards, window=20)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, alpha=0.3, label="Episode reward")
    if len(smoothed) > 0:
        offset = len(timesteps) - len(smoothed)
        plt.plot(
            timesteps[offset:],
            smoothed,
            label="Moving average (20 episodes)",
            linewidth=2,
        )
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Progress on LunarLander-v3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PATH)
    print(f"Learning curve saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
