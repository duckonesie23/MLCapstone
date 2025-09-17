import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

environmentStr = "CartPole-v1"
checkpoint_callback=CheckpointCallback(
            save_freq=1000,
            save_path="./logs/",
            name_prefix="ppo_"+environmentStr.lower()
        )  

def make_env(env_id: str):
    env = gym.make(env_id, render_mode="human")
    return env
    
def make_model(timesteps: int,env):
    if os.path.exists(f"logs/ppo_{environmentStr.lower()}.zip"):
        print("Loading existing model")
        model = PPO.load(f"logs/ppo_{environmentStr.lower()}.zip",env=env)
    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    return model

def train_model(model,env):
    episode_rewards = []
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action,_ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            episode_rewards.append(total_reward)
            print(f"Episode {len(episode_rewards)} reward: {total_reward}")
            total_reward = 0
            obs, _ = env.reset()
    env.close()


env=make_env(environmentStr)
model = make_model(25000,env)
model.save(f"logs/ppo_{environmentStr.lower()}.zip")



