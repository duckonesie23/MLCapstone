from select import select
from ssl import VERIFY_X509_TRUSTED_FIRST
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

environments = ["CartPole-v1","MountainCar-v0","LunarLander-v3","BipedalWalker-v3","CarRacing-v3"]

filelist = os.listdir(f"logs")
model = None
env = None

def menu():
    print("Which model would you like to view?")
    for i in range(len(environments)):
        print(f"{i}: {environments[i]}" )
    listModels(int(input()))


def returnLength(e):
    return len(e)

def listModels(selection):
    global env
    env = gym.make(environments[selection],render_mode = "human")

    modelVersions = []
    print(filelist)
    try:
        for file in filelist:
            if environments[selection].lower() in file:
                modelVersions.append(file)
        modelVersions.sort(key=returnLength)
        for i in range(len(modelVersions)):
            print(f"{i}: {modelVersions[i]}")
        loadModel(modelVersions[int(input())])
    except Exception as e:
        print(e)

def loadModel(versionName):
    global model 
    model = PPO.load(f"logs/{versionName}",env=env)
    
def replay(env,model):
    episode_rewards = []
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(1000):
        action,_ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            episode_rewards.append(total_reward)
            print(f"Episode {len(episode_rewards)} reward: {total_reward}")
            total_reward = 0
            obs, _ = env.reset()

menu()
replay(env,model)

'''env = gym.make(environments[1], render_mode="human")
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

try:
        for file in filelist:
            if environments[selection].lower() in file:
                modelVersions.append()
    except:
        print("error!")
        '''