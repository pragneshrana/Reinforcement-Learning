'''
This code will useful to simulate the bandit problem using different algorithm 
Epsilon greedy, softmax, UBC1
'''
import random
import matplotlib.pyplot as plt
import numpy as np

class simulate():
	'''
	This class will solve the bandit problem by simulating it using differnet algorithm
	'''

	def __init__(self,no_of_bandit_problem,no_of_arms,no_of_plays,mu,sigma,epsilon=0,temp=0,controls=0):
		'''
		Initilization of variables

		Input: 
		no_of_bandit_problem  : differnet bandit problem
		no_of_arms  : arms in each bandit problem
		no_of_plays : no of times bandit problem 
		'''
		self.no_of_bandit_problem = no_of_bandit_problem # number of bandit problems
		self.no_of_arms = no_of_arms # number of arms in each bandit problem
		self.no_of_plays = no_of_plays # number of times to pull each arm
		self.epsilon = epsilon	#(1-epsilon) times being greedy exploit and other time exploration
		self.mu = mu
		self.sigma = sigma
		self.temp = temp
		self.controls = controls

		#for simulator
		self.mean_assigned_to_arms = np.zeros([self.no_of_bandit_problem,self.no_of_arms])
		self.optimal_arm = np.zeros(no_of_bandit_problem)
		self.estimates=np.zeros([self.no_of_bandit_problem,self.no_of_arms])
		# reward estimated
		self.counter_of_pulled_arm=np.ones([self.no_of_bandit_problem,self.no_of_arms]) # number of times each arm was pulled # each arm is pulled atleast once
		
	def epsilon_greedy_policy(self,pulled_arm_reward,opt_arm_pull):
		'''
		part of epsilon greedy policy
		'''
		for i in range(self.no_of_bandit_problem) : 	
			if (random.random()<self.epsilon): 
				arm_index=np.random.randint(self.no_of_arms)											
			else : 
				arm_index=np.argmax(self.estimates[i])

			if (arm_index==self.optimal_arm[i]): # optimal action
				opt_arm_pull += 1

			each_reward=np.random.normal(self.mean_assigned_to_arms[i][arm_index],self.sigma)
			pulled_arm_reward.append(each_reward)
			self.counter_of_pulled_arm[i][arm_index] += 1
			self.estimates[i][arm_index] += (each_reward-self.estimates[i][arm_index])/self.counter_of_pulled_arm[i][arm_index]
		return pulled_arm_reward, opt_arm_pull 
	
	def softmax_policy(self,pulled_arm_reward,opt_arm_pull):
		'''
		part of epsilon softmax policy
		'''
		for i in range(self.no_of_bandit_problem) : 
			numerator=np.exp(self.estimates[i]/self.temp)
			probable=numerator/np.sum(numerator) # softmax probabilities
			arm_index=np.random.choice(range(self.no_of_arms),1,p=probable) # picks one arm based on softmax probability

			if (arm_index==self.optimal_arm[i]) : # To calculate % optimal action
				opt_arm_pull += 1
			
			each_reward=np.random.normal(self.mean_assigned_to_arms[i][arm_index],self.sigma)
			pulled_arm_reward.append(each_reward)
			self.counter_of_pulled_arm[i][arm_index] += 1
			self.estimates[i][arm_index] += (each_reward-self.estimates[i][arm_index])/self.counter_of_pulled_arm[i][arm_index]
		return pulled_arm_reward, opt_arm_pull 
	
	def ucb1(self,pulled_arm_reward,opt_arm_pull,pulled):
		'''
		simulate with upper confidence bound policy 
		'''
		for i in range(self.no_of_bandit_problem) : 
			ucb_Q=self.estimates[i]+np.sqrt(self.controls*np.log(pulled)/self.counter_of_pulled_arm[i])
			arm_index=np.argmax(ucb_Q)

			if (arm_index==self.optimal_arm[i]): # optimal action
				opt_arm_pull += 1

			each_reward=np.random.normal(self.mean_assigned_to_arms[i][arm_index],self.sigma)
			pulled_arm_reward.append(each_reward)
			self.counter_of_pulled_arm[i][arm_index] += 1
			self.estimates[i][arm_index] += (each_reward-self.estimates[i][arm_index])/self.counter_of_pulled_arm[i][arm_index]
		return pulled_arm_reward, opt_arm_pull 

	def bandit_simulation(self, simulator = 'epsilon_greedy'):
		'''
		This method will simulate the bandit problem using different al values 
		It shows what is ideal way to do tradeoff between exploration and explotation
		'''
		self.mean_assigned_to_arms=np.random.normal(self.mu,self.sigma,(self.no_of_bandit_problem,self.no_of_arms)) # generating the true means q*(a) for each arm for all bandits
		self.optimal_arm=np.argmax(self.mean_assigned_to_arms,1) # actual optimal arms for each bandit #row wise vandit

		# Pull all arms once
		first_reward=np.random.normal(self.mean_assigned_to_arms,1) # initial pulling of all arms

		avg_reward=[]
		avg_reward.append(np.mean(first_reward))	
		optimal_actions=[] #av
		optimal_actions.append(0) #first optimal action to balance

		for i in range(1,self.no_of_plays) :  
			'''
			first play-> over all m/c -> second play -> over all m/c -> .... -> over all m/c
			'''
			pulled_arm_reward=[] # all reward in 
			opt_arm_pull=0 # number of pulled of best arm in this time step

			if(simulator == 'epsilon_greedy'):
				pulled_arm_reward, opt_arm_pull = self.epsilon_greedy_policy(pulled_arm_reward,opt_arm_pull)
			if(simulator == 'softmax'):
				pulled_arm_reward, opt_arm_pull = self.softmax_policy(pulled_arm_reward,opt_arm_pull)
			if(simulator == 'ucb1'):
				pulled_arm_reward, opt_arm_pull = self.ucb1(pulled_arm_reward,opt_arm_pull,i)
			avg_R_pull=np.mean(pulled_arm_reward)		
			avg_reward.append(avg_R_pull)
			optimal_actions.append(float(opt_arm_pull)*100/self.no_of_bandit_problem)

		return avg_reward,optimal_actions
	
	def plotiing(self,avg_reward,optimal_Action,epsilon):
		'''
		ploting of epsilon greedy technique
		'''

		for i in range(len(epsilon)):	
			plt.plot(range(self.no_of_plays),avg_reward[i],label = "$\epsilon=$"+str(epsilon[i]))
			plt.rc('text',usetex=True)
			plt.title(r'$\epsilon$-greedy : Avg Reward Vs Steps')
			plt.ylabel('Avg Reward')
			plt.xlabel('Steps')
			plt.legend(loc='best')
		plt.savefig('reward_epsilon.jpg')
		plt.show()
		
		for i in range(len(epsilon)):	
			plt.plot(range(self.no_of_plays),optimal_Action[i],label ="$\epsilon=$"+str(epsilon[i]))
			plt.title(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps')
			plt.ylabel(r'$\%$ Optimal Action')
			plt.xlabel('Steps')
			plt.ylim(0,100)
			plt.legend(loc='best')
		plt.savefig('optimal_epsilon.jpg')
		plt.show()

if __name__ == '__main__':
	no_of_bandit_problem=2000 # number of bandit problems
	no_of_arms=1000 # number of arms in each bandit problem
	no_of_plays=1000 # number of times to pull each arm

	reward_result = []
	optimal_action_result = []
	epsilon = [0,0.02,0.1,1]
	temp =[0.01,1,1,10]
	controls = [0.1,2,5]

	choice= ['epsilon_greedy','softmax','ucb1']

	#case epsilon = 0
	epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,epsilon=epsilon[0],mu=0,sigma=1)
	reward_0 , optimal_action_0 = epsilon_greedy_simulation.bandit_simulation('epsilon_greedy')
	reward_result.append(reward_0)
	optimal_action_result.append(optimal_action_0)

	#case epsilon = 0.02
	epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,epsilon=epsilon[1],mu=0,sigma=1)
	reward_1 , optimal_action_1 = epsilon_greedy_simulation.bandit_simulation('epsilon_greedy')
	reward_result.append(reward_1)
	optimal_action_result.append(optimal_action_1)

	#case epsilon = 0.1
	epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,epsilon=epsilon[2],mu=0,sigma=1)
	reward_2 , optimal_action_2 = epsilon_greedy_simulation.bandit_simulation('epsilon_greedy')
	reward_result.append(reward_2)
	optimal_action_result.append(optimal_action_2)

	#case epsilon = 1
	epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,epsilon=epsilon[3],mu=0,sigma=1)
	reward_3 , optimal_action_3 = epsilon_greedy_simulation.bandit_simulation('epsilon_greedy')
	reward_result.append(reward_3)
	optimal_action_result.append(optimal_action_3)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=temp[0],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('softmax')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=temp[1],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('softmax')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=temp[2],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('softmax')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=temp[3],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('softmax')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=controls[0],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('ucb1')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=controls[1],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('ucb1')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)

	# #case softmax = 1
	# epsilon_greedy_simulation = simulate(no_of_bandit_problem=no_of_bandit_problem,no_of_arms = no_of_arms ,no_of_plays = no_of_plays,temp=controls[2],mu=0,sigma=1)
	# reward , optimal_action = epsilon_greedy_simulation.bandit_simulation('ucb1')
	# reward_result.append(reward)
	# optimal_action_result.append(optimal_action)



	#plotting
	epsilon_greedy_simulation.plotiing(reward_result,optimal_action_result,epsilon)

	plt.show()
