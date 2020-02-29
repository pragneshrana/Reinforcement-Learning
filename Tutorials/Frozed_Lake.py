import numpy as np
import matplotlib.pyplot as plt
import gym

#set the envirnment 
env = gym.make('FrozenLake-v0')


n_game = 1000	#total number of episodes 
win_pct = []	#win percentage
scores = []	#score after each play 

for i in range(n_game):
	done = False
	obs = env.reset()	
	score = 0
	while not done:
		'''
		'''
		action = env.action_space.sample() #ramdon action

		'''
		observation : 
		return  - object
		It's an environment specific object.
		which represents observation of the environment 
		ex: board states in game

		reward : 	
		return - float 
		Amount of reward achieved by the previous action.
		Goal is to increase the total reward

		done:
		return  - boolean
		True - episodes ended

		info:
		return - dictionary 
		useful for debugging
		contains some raw probabilites		
		'''
		obs, reward, done, info = env.step(action)

		'''
		'''
		score += reward #score==return

	scores.append(score)

	if i % 10 == 0: #after every 10 games keep track of reward 
		average = np.mean(scores[-10:])
		win_pct.append(average)

plt.plot(win_pct)
plt.show()

