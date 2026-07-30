"""
test.py
Renders a live visual window showing the trained PPO agent
piloting the lander on LunarLander-v3.
"""

import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_lunarlander"
N_EPISODES = 5


def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    for episode in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()

        print(f"Episode {episode}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
