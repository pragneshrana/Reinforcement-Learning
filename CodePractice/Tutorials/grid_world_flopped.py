import numpy as np
import matplotlib.pyplot as plt


#class agent 

class grid: #enviornment
	def __init__ (self,width,height,start):
		self.width = width
		self.height = height
		self.i = start[0]	#2d 
		self.j = start[1]	#2d
	
	def set(self,reward,actions):
		#reward should be a  dict of: (i,j): r(row,col) : reward
		#reward should be a  dict of: (i,j): r(row,col) : reward
		self.rewards = rewards
		self.actions = actions

	def set_state(self,s):
		self.i = s[0]
		self.j = s[1]
	
	def current_state(self):
		return (self.i,self.j)
	
	def is_terminal(self,s):
		return s not in self.actions
	
	def move(self,action):
		#check if legal move first 
		if action in self.actions[(self.i,self.j)]:
			if action == 'U' :
				self.i -= 1 
			if action == 'D' :
				self.i += 1
			if action == 'R' :
				self.i += 1
			if action == 'L' :
				self.i -= 1

		return self.reward.get((self.i,self.j),0)

	def undo_move(self,action):
		#opposite to above 
		if action == 'U' :
			self.i -= 1
		if action == 'D' :
			self.i += 1
		if action == 'R' :
			self.i += 1
		if action == 'L' :
			self.i -= 1
		#raise as exception if we arrive somewhere we shouldn't be
		
		assert(self.current_state() in self.all_states())

	def game_over(self):
		#return tue if game is over else false
		#true if we are in state where noa ction are possible
		return (self.i,self,j) not in self.actions
	
	def all_states(self):
		#possibly buggy but sumple way to get all the states 
		#either a position that has possible next actions
		#or a position that yields a reward
		return self(self.actions.keys() + self.reward.keys())		

def standard_grids():
	'''
	define a grid taht describes the reward for arriving at each state
	and possible actions at each state
	the grid looks like this 
	x means you can't go therer 
	s means start the positions 
	number means reward  st that state 
	.  .  .  1
	.  x  . -1
	s  .  .  .
	'''
	g =grid (3,4,(2,0))
	reward = {(0,3):1,(1,3):-1}
	actions = {
	(0,0):('D','R'),
	(0,1):('L','R'),
	(0,2):('L','D','R'),
	(1,0):('U','R'),
	(1,2):('U','D','R'),
	(2,0):('U','R'),
	(2,1):('L','R'),
	(2,2):('L','R','U'),
	(2,3):('L','U'),
	}
	
	g.set(reward,actions)
	return g

def negative_grid(step_cost = -0.1):
	#in this game we want to try to minimize the number of moves 
	#so we will penalize the every move
	g = standard_grid()
	g.rewrads.update({
	(0,0):step_cost ,
	(0,1):step_cost ,
	(0,2):step_cost ,
	(1,0):step_cost ,
	(1,2):step_cost ,
	(2,0):step_cost ,
	(2,1):step_cost ,
	(2,2):step_cost ,
	(2,3):step_cost ,
	})
	return g

def play_game(agent,env):
	pass
	














