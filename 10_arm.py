import random
import numpy as np
import matplotlib.pyplot as plt
class bandit_simulator():

  def __init__(self,no_of_arms,mu,sigma,no_of_plays,no_of_bandit_prob,epsilon=0,temperature=0):
    '''
    Initializing the variables
    '''
    #for one bandit problem or 2000 diff bandit problem
    self.no_of_arms = no_of_arms  #number of arams
    self.mu = mu #given mean for distribution of arms
    self.sigma = sigma #given sigma for distribution of arms
    self.arms = None #Arms assigned for specified bandit problem
    self.epsilon = epsilon #epsilon greedy 
    self.no_of_plays = no_of_plays #given plays in each round #number of steps
    self.avg_reward = np.zeros(no_of_plays)
    self.no_of_bandit_prob = no_of_bandit_prob #different type of bandit problems

    #for specific bandit problem or 1000 play
    self.reward = np.zeros(no_of_arms) #avg reward for all arm 
    self.selected_arm_index = None #arm index of selected arm
    self.selected_arm_mean = None #assign mean of selected arm  
    self.selected_arm_sigma = 1
    self.reward_after_each_play = None #reward after each play
    self.counter_of_selected_arm = np.zeros(no_of_arms)

    #for softmax
    self.softmax_converted_reward = np.zeros(no_of_arms)
    self.temperature = temperature
  def distribution_to_each_arm(self):
    '''
    Assinging the random mean to each arms
    '''
    self.arms = np.random.normal(self.mu, self.sigma, self.no_of_arms) #mean assigned to every arm
    # print('self.arms: ', self.arms)
    # print('print: ', print)
    

  def intial_reward(self):
    '''
    Intialize every arm with zero reward
    '''
    self.reward = np.zeros(self.no_of_arms) 
  
  def random_pick(self):
    '''
    as all the arms intialized zero reward so first arm will be randomly  picked up
    or
    Every arm has same reward then pick random arm
    or
    epsilon times it will arm randomly
    '''
    self.selected_arm_index = random.choice(range(self.no_of_arms))
    self.selected_arm_mean = self.arms[self.selected_arm_index]
  
  def obtain_reward(self):
    '''
    once arm is picked obtain reward from specified distribution of picked arm
    '''
    self.reward_after_each_play = np.random.normal(self.selected_arm_mean, self.selected_arm_sigma,1)
  
  def update_counter_of_selected_arm(self):
    '''
    update counter for selected arms
    '''
    self.counter_of_selected_arm[self.selected_arm_index] += 1 
    
  
  def update_reward(self):
    '''
    This methos will update reward after each play
    '''
    count = self.counter_of_selected_arm[self.selected_arm_index] #just to reduce size of eqaution
    self.reward[self.selected_arm_index] = (self.reward[self.selected_arm_index]*(count-1)+ self.reward_after_each_play)/count

  def epsilon_greedy_arm_pick(self):
    '''
    It will pick the arm which has highest reward in previous rum
    '''
    if(random.random() > (1-self.epsilon)):
      self.random_pick()
    else:
      self.selected_arm_index = np.argmax(self.reward)
      # print('self.reward: ', self.reward)
      # print(self.selected_arm_index)
      self.selected_arm_mean = np.max(self.arms)
      # print(self.selected_arm_mean)
  
  
  def softmax_action_pick(self):
    '''
    softmax action follows gibbs distribution. Apply the gibbs distribution(softmax) previous reward it will convert the output into the 
    probability and it will pick the arm based on supplied probability.
    '''
    numerator = np.exp(self.reward/self.temperature)
    sum_soft_reward =np.sum(numerator) #denomenator
    self.softmax_converted_reward = numerator/sum_soft_reward
    
    ###Pick arm based on distribution
    # draw = choice(list_of_candidates, number_of_items_to_pick,p=probability_distribution)
    self.selected_arm_index = np.random.choice([x for x in range(self.no_of_arms)],1,p=self.softmax_converted_reward)[0]
    self.selected_arm_mean = self.arms[self.selected_arm_index]

  def avg_reward_update(self,play_index):
    '''
    This method will update the avg reward for diffrent type of bandit problem
    '''
    self.avg_reward[play_index] +=  self.reward_after_each_play
  
  def check_result_greedy(self):
    '''
    chekc code by printing output
    '''
    print('Mean assigned to each arm',self.arms)
    print('Arm picked:',self.selected_arm_index)
    print('Reward_after_each_play :',self.reward_after_each_play)
    print('Rewads with each arm',self.reward)
    print('Avg reward :',self.avg_reward)
    print('counter_of_selected_arm: ', self.counter_of_selected_arm)
    print('self.selected_arm_mean: ', self.selected_arm_mean)
    print('\n')
  
  def check_result_softmax(self):
    '''
    chekc code by printing output
    '''
    print('Mean assigned to each arm',self.arms)
    print('self.selected_arm_index: ', self.selected_arm_index)
    print('Reward_after_each_play :',self.reward_after_each_play)
    print('softmax_converted_reward: ', self.softmax_converted_reward)
    print('Rewads with each arm',self.reward)
    print('Avg reward :',self.avg_reward)
    print('counter_of_selected_arm: ', self.counter_of_selected_arm)
    print('self.selected_arm_mean: ', self.selected_arm_mean)
    print('\n')

  def simulate_epsilon_greedy_policy(self):
    '''
    This method simulates.
    It will pick the arm which has highest overall reward in previous plays.
    '''
    for i in range(self.no_of_bandit_prob):
      print('Bandit problem: ', i)
      self.distribution_to_each_arm()
      self.intial_reward()
      #zeroth play
      self.random_pick()
      self.obtain_reward()
      self.update_counter_of_selected_arm()
      self.update_reward()
      self.avg_reward_update(0)
      # self.check_result_greedy()
      for j in range(1,self.no_of_plays):
        self.epsilon_greedy_arm_pick()
        self.obtain_reward()
        self.update_counter_of_selected_arm()
        self.update_reward()
        self.avg_reward_update(j)
        # self.check_result_greedy()
    final_avg_reward = self.avg_reward/self.no_of_bandit_prob
    return final_avg_reward

  def simulate_softmax_policy(self):
    '''
    This method simulates based on gibbs distribution and supplied .
    It will pick the arm which has highest overall reward in previous plays.
    '''
    for i in range(self.no_of_bandit_prob):
      print('Bandit problem: ', i)
      self.distribution_to_each_arm()
      self.intial_reward()
      #zeroth play
      self.random_pick()
      self.obtain_reward()
      self.update_counter_of_selected_arm()
      self.update_reward()
      self.avg_reward_update(0)
      # self.check_result_softmax()
      for j in range(1,self.no_of_plays):
        self.softmax_action_pick()
        self.obtain_reward()
        self.update_counter_of_selected_arm()
        self.update_reward()
        self.avg_reward_update(j)
        # self.check_result_softmax()
    final_avg_reward = self.avg_reward/self.no_of_bandit_prob
    return final_avg_reward

print('hello')
no_of_plays = 1000
no_of_bandits = 200

greedy = bandit_simulator(no_of_arms=10,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,epsilon=0)
avg_reward_zero_greedy = greedy.simulate_epsilon_greedy_policy()

greedy_2 = bandit_simulator(no_of_arms=10,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,epsilon=0.001)
avg_reward_01_greedy = greedy.simulate_epsilon_greedy_policy()

greedy_3 = bandit_simulator(no_of_arms=10,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,epsilon=0.01)
avg_reward_1_greedy = greedy.simulate_epsilon_greedy_policy()

# softmax_05 = bandit_simulator(no_of_arms=3,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,temperature=0.1)
# avg_reward_soft_05 = softmax_05.simulate_softmax_policy()
 
# softmax_1 = bandit_simulator(no_of_arms=10,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,temperature=1)
# avg_reward_soft_1= softmax_1.simulate_softmax_policy()

# softmax_50 = bandit_simulator(no_of_arms=10,mu=0,sigma=1,no_of_plays=no_of_plays,no_of_bandit_prob=no_of_bandits,temperature=500)
# avg_reward_soft_50 = softmax_50.simulate_softmax_policy()

import matplotlib.pyplot as plt
# plt.plot(avg_reward_soft_05)
# plt.plot(avg_reward_soft_1)
# plt.plot(avg_reward_soft_50)
plt.plot(avg_reward_zero_greedy)
plt.plot(avg_reward_01_greedy)
plt.plot(avg_reward_1_greedy)
plt.xlabel('No of plays of each bandit problem')
plt.ylabel('Avg Reward')
plt.show()