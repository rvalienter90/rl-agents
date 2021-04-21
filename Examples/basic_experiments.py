import gym
import highway_env

env = gym.make("highway-v0")
done = False
while not done:
    action = 3 #"FASTER"# Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()


"""
arrows
up - lane left 
down - lane right
right - accelerate
left - decelerate 

"""
env = gym.make("highway-v0")
env.configure({
    "manual_control": True
})
env.reset()
done = False
while not done:
    env.step(env.action_space.sample())  # with manual control, these actions are ignored
    env.render()