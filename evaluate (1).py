"""
evaluate.py
Loads the saved PPO model and runs deterministic evaluation episodes
on LunarLander-v3, printing per-episode scores and a summary of the
mean/std performance.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = "models/ppo_lunarlander"
N_EVAL_EPISODES = 20
N_SHOW_EPISODES = 5  # number of individual episode scores to print


def rating(mean_score):
    if mean_score >= 200:
        return "Excellent"
    if mean_score >= 100:
        return "Good"
    if mean_score >= 0:
        return "Fair"
    return "Needs improvement"


def main():
    env = gym.make("LunarLander-v3")
    model = PPO.load(MODEL_PATH, env=env)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    print(f"Deterministic Mean Evaluation Score ({N_EVAL_EPISODES} Episodes): "
          f"{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Rating: {rating(mean_reward)}\n")

    print(f"Sample individual episode scores (first {N_SHOW_EPISODES}):")
    for ep in range(1, N_SHOW_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"  Episode {ep}: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
