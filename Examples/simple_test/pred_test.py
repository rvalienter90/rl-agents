import gym
import highway_env
import numpy as np
from gym.wrappers import Monitor
# env = gym.make("highway-modif-v0")
env = gym.make("intersection-modif-v0")
# env = gym.make("intersection-v0")

env.reset()
# env = Monitor(env, directory="highway_pred/videos", video_callable=lambda e: True)
# env.set_monitor(env)
# env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
done = False
# while True
env.reset()
action = 1

# ACTIONS_LONGI = {
#     0: 'SLOWER',
#     1: 'IDLE',
#     2: 'FASTER'
# }
while not done:

    # action = np.random.randint(0,3) # Your agent code here
    action = 0 if action ==2 else 2
    obs, reward, done, info = env.step(action)
    env.render()


env.close()




