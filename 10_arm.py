import random
import numpy as np
import matplotlib.pyplot as plt
class epsilon_greedy_algo():

  def __init__(self,no_of_arms,mu,sigma,epsilon,no_of_plays,no_of_bandit_prob):
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
    self.reward_after_each_play = np.random.normal(self.selected_arm_mean, self.selected_arm_sigma)
  
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
      self.selected_arm_mean = np.max(self.reward_after_each_play)
      # print(self.selected_arm_mean)
  
  def avg_reward_update(self,play_index):
    '''
    This method will update the avg reward for diffrent type of bandit problem
    '''
    self.avg_reward[play_index] +=  self.reward_after_each_play
  
  def chekc_result(self):
    '''
    chekc code by printing output
    '''
    print('Arm picked:',self.selected_arm_index)
    print('Reward_after_each_play :',self.reward_after_each_play)
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
      # self.chekc_result()
      for j in range(1,self.no_of_plays):
        self.epsilon_greedy_arm_pick()
        self.obtain_reward()
        self.update_counter_of_selected_arm()
        self.update_reward()
        self.avg_reward_update(j)
        # self.chekc_result()
    final_avg_reward = self.avg_reward/self.no_of_bandit_prob
    return final_avg_reward

print('hello')
no_of_plays = 1000
greedy = epsilon_greedy_algo(no_of_arms=3,mu=0,sigma=1,epsilon=0,no_of_plays=no_of_plays,no_of_bandit_prob=2000)
avg_reward = greedy.simulate_epsilon_greedy_policy()
print(len(avg_reward))
import matplotlib.pyplot as plt
plt.plot(avg_reward)
plt.xlabel('No of plays of each bandit problem')
plt.ylabel('Avg Reward')
plt.show()