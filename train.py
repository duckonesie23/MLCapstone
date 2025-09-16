from re import L
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

environmentStr = "CartPole-v1"

def make_env(env_id: str):
    def init():
        env = gym.make(env_id, render_mode="human")
        checkpoint_callback=CheckpointCallback(
            save_freq=1000,
            save_path="./logs/",
            name_prefix="ppo_"+env_id.lower()
        )  
        return env
    return init

def make_model(timesteps: int):
    model = PPO("MlpPolicy",make_env(env))
    model.learn(total_timesteps=25000)
    return model


env=make_env()

obs = env.reset()
