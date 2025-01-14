import gym
# env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, "recording", force=True)
# env.reset()
# while True:
#     obs, rew, done, info = env.step(env.action_space.sample())
#     if done:
#         break


env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "./vid", force=True)
env.seed(0)
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()  # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()