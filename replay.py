import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

environments = ["CartPole-v1","MountainCar-v0","LunarLander-v3","BipedalWalker-v3","CarRacing-v3"]

env = gym.make(environments[1], render_mode="human")
model = PPO.load(f"logs/ppo_{environments[1]}.zip",env=env)

episode_rewards = []
obs, _ = env.reset()
total_reward = 0

for _ in range(5):
    action,_ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        episode_rewards.append(total_reward)
        print(f"Episode {len(episode_rewards)} reward: {total_reward}")
        total_reward = 0
        obs, _ = env.reset()