import gymnasium as gym
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")

if os.path.exists("ppo_cartpoleB.zip"):
    print("loading")
    model = PPO.load("ppo_cartpoleB",env=env)
else:
    print("training")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpoleB_tensorboard/")

model.learn(total_timesteps=10000)

episode_rewards = []
obs, _ = env.reset()
total_reward = 0

for _ in range (1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done or truncated:
        episode_rewards.append(total_reward)
        print(f"Episode {len(episode_rewards)} reward: {total_reward}")
        total_reward = 0
        obs, _ = env.reset()
env.close()
model.save("ppo_cartpoleB")

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reard")
plt.title("Learning Curve")
plt.show()