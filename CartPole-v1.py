import gymnasium as gym
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

#cartpole
#CartPole-v1, LunarLander-v3, Pendulum-v1

modelStr = "pong"
environmentStr = "Pong-v4"

env = gym.make(environmentStr, render_mode="human")

if os.path.exists(f"ppo_{modelStr}.zip"):
    print("loading")
    model = PPO.load(f"ppo_{modelStr}",env=env)
else:
    print("training")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")

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
model.save(f"ppo_{modelStr}")

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reard")
plt.title("Learning Curve")
plt.show()