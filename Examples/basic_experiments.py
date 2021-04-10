import gym
import highway_env

env = gym.make("highway-v0")
done = False
while not done:
    action = 3 #"FASTER"# Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
