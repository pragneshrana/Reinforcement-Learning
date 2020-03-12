# Ref: code is taken from the source  and modified as per requirement 
# https://github.com/opocaj92/GridWorldEnvs 

#calling different library 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import math 
import os
import sys

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
from PIL import Image as Image

#Actions defined from
move_up = 0
move_right = 1
move_down = 2
move_left = 3

# defining colours for the plotting 
colours = {  0 : [1.0, 1.0, 1.0], 
             4 : [0.5, 0.5, 0.5],
            -1 : [0.0, 0.0, 1.0], 
            -2 : [0.0, 1.0, 0.0],
            -3 : [1.0, 0.0, 0.0], 
             3 : [1.0, 0.0, 1.0],
            -13: [0.5, 0.0, 0.5], 
              1: [0.0, 0.0, 0.0], 
             10:[1.0, 0.6, 0.4]  
          }


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, terminal_reward=10.0,move_reward=0, puddle1_reward=-1.0, puddle2_reward=-2.0, puddle3_reward=-3.0, westerly_wind=True):
        '''
        Initializing the variables 
        '''
        #map dimentions
        self.n = None
        self.m = None

        self.puddle1 = []
        self.puddle2 = []
        self.puddle3 = []
        self.walls = []
        self.goals = []
        self.start = []
        self.mapFile = None
        self.saveFile = False

        self.move_reward = move_reward
        self.puddle1_reward = puddle1_reward
        self.puddle2_reward = puddle2_reward
        self.puddle3_reward = puddle3_reward
        self.westerly_wind = westerly_wind
        self.terminal_reward = terminal_reward
        self.steps = 0
        self.discounted_return = 0
        self.figurecount = 0
        self.figtitle = None
        self.done = False
        self.opt_policy = None
        self.draw_arrows = False
        self.first_time = True

    #def _init
    def load_map(self, map_file):
        '''
        loading the map and storing into the  
        '''
        #rading file 
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_name = os.path.join(file_path, map_file)
        #reading the file 
        with open(self.file_name, "r") as f:
            for i, lines in enumerate(f):
                lines = lines.rstrip('\r\n')
                lines = lines.split(' ')
                if self.n is not None and len(lines) != self.n:
                    raise ValueError(
                        "Map's liness are not of the same dimension...")
                self.n = len(lines)
                for line_index , actual_state in enumerate(lines):
                    if   actual_state   == "4":
                        self.start.append([i, line_index])  # self.n * i + line_index
                    elif actual_state == "3":
                        self.goals.append([i, line_index])
                    elif actual_state == "-1":
                        self.puddle1.append([i, line_index])
                    elif actual_state == "-2":
                        self.puddle2.append([i, line_index])
                    elif actual_state == "-3":
                        self.puddle3.append([i, line_index])
                    elif actual_state == "1":
                        self.walls.append([i, line_index])
            self.m = i + 1
        if len(self.goals) == 0:
            raise ValueError("At least one goal needs to be specified...")
        self.n_states = self.n * self.m  #total number of states 
        self.n_actions = 4 #number of action 
        state_index = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0] #picking state 
        self.state = self.start[state_index] #start state 
        self.action_space = spaces.Discrete(4)  #It will return four possiblea actions
        self.observation_space = spaces.Discrete(self.n_states) #observation space based on
        self.observation = self.observations_from_map()

    def _step(self, action):
        '''
        calculating parameters after each steps and new states
        '''
        assert self.action_space.contains(action)
        if self.state in self.goals:  # Checking if reached goal or not
            return self.state, 0.0, self.done, None
        else:
            # Taking correct action with 0.9 probability
            if np.random.binomial(1, 0.9, 1)[0] == 1:
                new_state = self._take_action(action)
            else:  # Taking other 3 action with 0.1/3 probability
                temp_action = [0, 1, 2, 3]
                temp_action.remove(action)
                action_taken = np.random.choice(3, 1, p=[1/3, 1/3, 1/3])[0]
                new_state = self._take_action(temp_action[action_taken])

            reward = self._get_reward(new_state)
            self.state = new_state

            # Westerly wind moving additional to east with 0.5 probability
            if self.westerly_wind and np.random.binomial(1, 0.5, 1)[0] == 1:
                new_state = self._take_action(1)
                reward += self._get_reward(new_state)
                self.state = new_state

            return self.state[0]*14 + self.state[1], reward, self.done, None

    def _reset(self):
        '''
        Resetting the whole environment by reading the file again  
        '''
        if self.first_time:
            self.load_map(self.mapFile)
        self.done = False
        state_index = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        self.state = self.start[state_index]
        return self.state[0]*14 + self.state[1]

    def _render(self, mode='human', close=False):
        '''
        Redeing and plotting the results
        '''
        if close:
            return

        self.observation = self.observations_from_map()
        img = self.observation
        fig = plt.figure(1, figsize=(10, 8), dpi=60, facecolor='w', edgecolor='k')
        fig.canvas.set_window_title(self.figtitle)
        plt.clf()
        plt.xticks(np.arange(0, 15, 1))
        plt.yticks(np.arange(0, 15, 1))
        plt.grid(True)
        plt.title("Start:Grey, Goal:Pink, Robot:Orange\n -1:Blue, -2:Green, -3:Red, Wall:Black\nWesterly wind: 0.5 Prob. Random State: 0.1 Prob.", fontsize=10)
        plt.xlabel("Steps: %d, Reward: %f" %
                   (self.steps + 1, self.discounted_return), fontsize=10)

        plt.imshow(img, origin="upper", extent=[0, 14, 0, 14])
        fig.canvas.draw()

        # Drawing arrows
        if self.draw_arrows:
            for k, v in self.opt_policy.items():
                y = 13-int(k/14)
                x = k % 14
                self._draw_arrows(x, y, v)
            plt.savefig(self.figtitle+"_Arrows.png")

        plt.pause(0.0001)  # 0.01

        # For creating video from tmp png files
        if self.saveFile:
            fname = '_tmp%05d.png' % self.figurecount
            plt.savefig(fname)
            plt.clf()

        return

    def _draw_arrows(self, x, y, direction):
        '''
        supporting file for rendering
        It will show the array in the plot
        '''

        if direction == move_up:
            x += 0.5
            dx = 0
            dy = 0.4
        if direction == move_down:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if direction == move_right:
            y += 0.5
            dx = 0.4
            dy = 0
        if direction == move_left:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0
        plt.arrow(x,  # x1
                  y,  # y1
                  dx,  # x2 - x1
                  dy,  # y2 - y1
                  facecolor='k',
                  edgecolor='k',
                  width=0.005,
                  head_width=0.4)

    def _take_action(self, action):
        '''
        shifting current state based on action taken
        '''
        row = self.state[0] 
        col = self.state[1] 
        if action == move_down and [row + 1, col] not in self.walls:
            row = min(row + 1, self.m - 1)
        elif action == move_up and [row - 1, col] not in self.walls:
            row = max(0, row - 1)
        elif action == move_right and [row, col + 1] not in self.walls:
            col = min(col + 1, self.n - 1)
        elif action == move_left and [row, col - 1] not in self.walls:
            col = max(0, col - 1)
        new_state = [row, col]
        return new_state

    def _get_reward(self, new_state):
        '''
        If stuck in puddle then giving negative reward  based on level of puddle 
        '''
        if new_state in self.goals:
            self.done = True
            return self.terminal_reward
        elif new_state in self.puddle1:
            return self.puddle1_reward
        elif new_state in self.puddle2:
            return self.puddle2_reward
        elif new_state in self.puddle3:
            return self.puddle3_reward
        return self.move_reward

    def _seed(self, seed=None):
        '''
        seeding the state 
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observations_from_map(self):
        '''
        Generating observations from the map 
        '''
        grid_map = self.read_gridworld()
        obs_shape = [14, 14, 3]
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    if [i, j] == self.state:
                        this_value = colours[10][k]
                    else:
                        this_value = colours[grid_map[i, j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)
                                * gs1, k] = this_value
        return observation

    def read_gridworld(self):
        '''
        Reading gridworld file
        '''
        grid_map = open(self.file_name, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

