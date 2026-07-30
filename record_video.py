"""
record_video.py
Wraps the environment with Gymnasium's RecordVideo wrapper and
records .mp4 flight footage of the trained agent to videos/.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_lunarlander"
VIDEO_DIR = "videos"
N_EPISODES = 3


def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix="rl-video",
    )

    model = PPO.load(MODEL_PATH)

    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Recorded episode {episode}: reward = {total_reward:.2f}")

    env.close()
    print(f"Videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
