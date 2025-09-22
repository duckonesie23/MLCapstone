import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

environments = ["CartPole-v1","MountainCar-v0","LunarLander-v3","BipedalWalker-v3"]

environmentStr = environments[3]
checkpoint_callback=CheckpointCallback(
            save_freq=1000,
            save_path="./logs/",
            name_prefix="ppo_"+environmentStr.lower()
        )  

def make_env(env_id: str):
    return gym.make(env_id, render_mode="human")
    
def make_model(env):
    if os.path.exists(f"logs/ppo_{environmentStr.lower()}.zip"):
        print("Loading existing model")
        model = PPO.load(f"logs/ppo_{environmentStr.lower()}.zip",env=env)
    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1)
    return model

def train_model(model,env,timesteps: int,render=True):
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    episode_rewards = []
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(5000):
        action,_ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
        if terminated or truncated:
            episode_rewards.append(total_reward)
            print(f"Episode {len(episode_rewards)} reward: {total_reward}")
            total_reward = 0
            obs, _ = env.reset()
    env.close()


env=make_env(environmentStr)
model = make_model(env)
train_model(model,env,1000)
model.save(f"logs/ppo_{environmentStr.lower()}.zip")

if __name__ == "__main__":
    # This only runs if the script is executed directly
    env = make_env(environmentStr)
    model = make_model(env)
    train_model(model, env, 2000)
    model.save(f"logs/ppo_{environmentStr.lower()}.zip")


