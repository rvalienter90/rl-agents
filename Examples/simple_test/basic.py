import gym
import highway_env

env = gym.make("highway-v0")
# env = gym.make("intersection-v0")

env.reset()
done = False
# while True
while not done:
    action = 1 # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()