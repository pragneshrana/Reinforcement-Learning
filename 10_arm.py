import random
import numpy as np
import matplotlib.pyplot as plt
class epsilon_greedy_algo():

  def __init__(self,no_of_arms,mu,sigma,epsilon,no_of_plays,no_of_bandit_prob):
    '''
    Initializing the variables
    '''
    self.no_of_arms = no_of_arms  #number of arams
    self.mu = mu #given mean for distribution of arms
    self.sigma = sigma #given sigma for distribution of arms
    self.arms = None #Arms assigned for specified bandit problem
    self.epsilon = epsilon #epsilon greedy 
    self.no_of_plays = no_of_plays #given plays in each round #number of steps
    self.avg_reward = np.zeros(no_of_plays)
    self.no_of_bandit_prob = no_of_bandit_prob #different type of bandit problems
    self.reward = np.zeros(no_of_arms) #avg reward for all arm 
    self.selected_arm_index = None #arm index of selected arm
    self.selected_arm_mean = None #assign mean of selected arm  
    self.selected_arm_sigma = 1
    self.reward_after_each_play = None #reward after each play

  def distribution_to_each_arm(self):
    '''
    Assinging the random mean to each arms
    '''
    self.arms = np.random.normal(self.mu, self.sigma, self.no_of_arms) #mean assigned to every arm

  def intial_reward(self):
    '''
    Intialize every arm with zero reward
    '''
    self.reward = np.zeros(no_of_arms) 
  
  def each_play(self):
    '''
    Each play of bandits
    In which it picks the arm with best reward based on previous reward records
    '''
    #pick an arm 
  
  def random_pick(self):
    '''
    as all the arms intialized zero reward so first arm will be randomly  picked up
    or
    Every arm has same reward then pick random arm
    or
    epsilon times it will arm randomly
    '''
    self.selected_arm_index = random.choice(range(len(self.no_of_arms)))
    self.selected_arm_mean = self.arms[arm_index]
  
  def greedy_arm_pick(self):
    '''
    It will pick the arm which has highest reward in previous rum
    '''
    self.selected_arm_index = 
    self.selected_arm_mean = 
  

  def obtain_reward(self):
    '''
    once arm is picked obtain reward from specified distribution of picked arm
    '''
    self.reward_after_each_play = np.random.normal(self.selected_arm_mean, self.selected_arm_sigma)
  
  def update_reward(self):
    '''
    This methos will update reward after each play
    '''
    self.reward[self.selected_arm_index] += self.reward_after_each_play
  
  def avg_reward_update(self,play_index):
    '''
    This method will update the avg reward for diffrent type of bandit problem
    '''
    self.avg_reward[play_index] +=  self.reward_after_each_play
  
  def simulate_epsilon_greedy_policy(self):
    '''
    This method simulates.
    It will pick the arm which has highest overall reward in previous plays.
    '''
    self.distribution_to_each_arm()


print('hello')
greedy = epsilon_greedy_algo(no_of_arms=5,mu=0,sigma=1,epsilon=0,no_of_plays=10,no_of_bandit_prob=10)
greedy.simulate()