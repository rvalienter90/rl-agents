import gym
from gym.wrappers import Monitor
import highway_env

import matplotlib.pyplot as plt
import time
import highway_env.grader as grader

# env = gym.make("highway-v0")
env = gym.make("intersection-v0")

env = Monitor(env, directory="run2", video_callable=lambda e: True)
#env.unwrapped.set_monitor(env)

obs = env.reset()
done = False

# while True
while not done:
    action = 1 # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()

