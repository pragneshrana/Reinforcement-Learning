#https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('info: ', info)
        print('done: ', done)
        print('reward: ', reward)
        print('observation: ', observation) #https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
