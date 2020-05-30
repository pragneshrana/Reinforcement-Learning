# The code red reference is given below:

# Some GridWorld environments for OpenAI Gym by Castellini Jacopo.
# Source : https://github.com/opocaj92/GridWorldEnvs

# Four room gird world environment is taken from Aharutyu git
# Source : https://github.com/aharutyu/gym-gridworlds/blob/master/gridworlds/envs/four_rooms.py

import numpy as np
from gym import wrappers
import dill
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict
import sys

class learn_option_policy:
    '''
    this class will learn option policy
    '''
    def __init__(self,alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val):
        self.alpha_val = alpha_val # As described in paper
        self.gamma_val = gamma_val
        self.epsilon_val = epsilon_val
        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        self.RENDERING_switch = RENDERING_switch
        pass
    
    # Creates epsilon_val greedy policy
    def epsilon_greedy_policy(self,Q, nA):
        def policy_function(obser):
            A = np.ones(nA, dtype=float) * self.epsilon_val / nA

            # Taking random action if all are same
            if np.allclose(Q[obser], Q[obser][0]):
                req_best_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            else:
                req_best_action = np.argmax(Q[obser])

            A[req_best_action] += (1.0 - self.epsilon_val)
            return A
        return policy_function

    # Q Learning algorithm implementation
    def q_learning(self,environment,alpha_val,Option=1):
        environment.reset()
        environment.epsilon_val = self.epsilon_val
        environment.gamma_val = self.gamma_val
        environment.alpha_val = alpha_val


        Q = defaultdict(lambda: np.zeros(environment.n_actions))

        number_of_steps = np.zeros(self.num_episodes)
        total_return = np.zeros(self.num_episodes)

        for iterate in range(self.num_iterations):

            Q.clear()

            policy = self.epsilon_greedy_policy(Q, environment.n_actions)
            figcount = 0
            options_goal = [[25, 56, 77, 103], [103, 25, 56, 77]]
            options_start_states = [[np.append(np.arange(0, 25, 1), 103), np.arange(25, 56, 1), np.arange(56, 77, 1), np.arange(77, 103, 1)],
                                    [np.arange(0, 26, 1), np.arange(
                                        26, 57, 1), np.arange(57, 78, 1), np.arange(78, 104, 1)]]

            for op in np.arange(4):

                for i_epi in range(self.num_episodes):
                    discounted_return = 0
                    steps_per_epi = 0

                    if (i_epi + 1) % 100 == 0:
                        print("\nIteration: {} Episode {}/{}.".format(iterate,
                                                                i_epi + 1, self.num_episodes))
                    #obser ~ observer
                    obser = environment.reset()  # Start state
                    environment.state = np.random.choice(options_start_states[Option][op])
                    environment.start_state = environment.state
                    environment.terminal_state = options_goal[Option][op]
                    obser = environment.state

                    for i in itertools.count():  # Till the end of episode
                        action_prob = policy(obser)
                        a = np.random.choice(
                            [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                        
                        next_observation, reward, done, _ = environment.step(a)  # Taking action

                        environment.steps_count = i
                        discounted_return += reward*self.gamma_val**i  # Updating return
                        environment.discounted_return = discounted_return
                        steps_per_epi += environment.options_length

                        if self.RENDERING_switch:  # Rendering
                            environment.render(mode='human')
                            # environment.figurecount = figcount
                            # figcount += 1

                        if (next_observation not in options_start_states[Option][op]) and (next_observation != options_goal[Option][op]) and not done:
                            # print("Outside the room")
                            break

                        # Finding next best action from next state
                        best_next_a = np.argmax(Q[next_observation])
                        # Q Learning update
                        Q[obser][a] += alpha_val*(reward + self.gamma_val* Q[next_observation][best_next_a] - Q[obser][a])

                        if done:
                            print("Total discounted return is :", discounted_return)
                            print("Total steps taken is :", i)
                            environment.discounted_return = 0
                            environment.steps_count = 0
                            break
                        obser = next_observation
                # print("Total steps taken is :", steps_per_epi)
                # Updating Number of steps
                number_of_steps[i_epi] += steps_per_epi
                # Updating return
                total_return[i_epi] += self.gamma_val**steps_per_epi

        number_of_steps /= self.num_iterations
        total_return /= self.num_iterations  # Updating return

        return Q, number_of_steps, total_return


    def plot_avg_steps(self, avg_steps, fig_save, prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, avg_steps)
        plt.title("SMDP Q-Learning Average steps per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of average steps per episode")
        plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[0] + prob_type + "_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        plt.ylim((0, 1000))
        plt.savefig(fig_save[1] + prob_type + "_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()


    def plot_total_return(self,total_return,fig_save, prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, total_return)
        plt.title("SMDP Q-Learning Average Return per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[2] + prob_type + "_episodes_" + str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()

    def generate_option_policy(self):
        '''
        this method will simulate the learn policy 
        '''
        # Setting some values for Option 1 & Option 2
        Option = 1
        if Option == 1:
            mapFiles = ["map1.txt", "map2.txt"]
            prob_type = ["G1", "G2"]
            filesave = ["O1.pkl", "O2.pkl"]
            fig_title = ["O1", "O2"]
            filesave_option = [["V_O1.pkl", "P_O1.pkl"],
                            ["V_O2.pkl", "P_O2.pkl"]]
            numpy_save = ['Q_saves/Q1_avg_num_of_steps_',
                        'Q_saves/Q1_avg_num_of_steps_']
            fig_save = ['Q1_Avg_steps_Prob_',
                        'Q1_Avg_steps_cut_Prob_', 'Q1_Return_Prob_']

        else:
            mapFiles = ["map3.txt", "map4.txt"]
            prob_type = ["G1", "G2"]
            filesave = ["O1.pkl", "O2.pkl"]
            filesave_option = [["Q2_G1_O1.pkl", "Q2_G1_O2.pkl"],
                            ["Q2_G2_O1.pkl", "Q2_G2_O2.pkl"]]
            fig_title = ["O1", "O2"]
            numpy_save = ['Q_saves/Q2_avg_num_of_steps_',
                        'Q_saves/Q2_avg_num_of_steps_']
            fig_save = ['Q2_Avg_steps_Prob_',
                        'Q2_Avg_steps_cut_Prob_', 'Q2_Return_Prob_']
        
        

        for pb in range(2):
            environment = gym.make("GridWorld-v0")
            environment.mapFile = mapFiles[pb]
            environment.saveFile = False
            environment.action_space = gym.spaces.Discrete(4)  # spaces.Discrete(4)
            environment.n_actions = 4
            environment.reset()
            environment.first_time = False
            # environment.draw_circles = True
            environment.figtitle = fig_title[pb]
            Option = pb

            Q, number_of_steps, total_return = self.q_learning(environment,self.alpha_val[pb],Option)

            dill.dump(Q, open(filesave[pb], 'wb'))

            np.save(numpy_save[pb]+prob_type[pb], number_of_steps)
            np.save(numpy_save[pb]+prob_type[pb], total_return)

            self.plot_avg_steps( number_of_steps, fig_save, prob_type[pb])
            self.plot_total_return(total_return, fig_save, prob_type[pb])

            dill.dump(Q, open(filesave[pb], 'wb'))

            # Finding V values
            Q = dill.load(open(filesave[pb], 'rb'))
            V = np.zeros(len(Q)) # For storing V values
            optimal_policy = np.zeros(len(Q),dtype='int8') # For storing optimal policy

            for j in range(len(Q)):
                V[j] = np.max(Q[j])
                optimal_policy[j] = np.argmax(Q[j])

            V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
            for g in [25, 56, 77, 103]:
                V[g] = 0
            if pb == 0: # Setting optimal action for hallway states according to option
                optimal_policy[25] = 1
                optimal_policy[56] = 2
                optimal_policy[77] = 3
                optimal_policy[103] = 0
            else:
                optimal_policy[25] = 3
                optimal_policy[56] = 0
                optimal_policy[77] = 1
                optimal_policy[103] = 2

            dill.dump(V, open(filesave_option[pb][0], 'wb'))
            dill.dump(optimal_policy, open(filesave_option[pb][1], 'wb'))

            environment.V = V # passing value function to environment 
            environment.optimal_policy = optimal_policy # passing optimal policy to environment
            environment.terminal_state = 1000
            environment.draw_circles = True
            environment.draw_arrows = False
            environment.figtitle = fig_title[pb]+'_'+str(self.num_episodes)
            environment.render(mode='human')
            environment.draw_circles = False
            environment.draw_arrows = True
            environment.render(mode='human')


class SMDP_Q():

    def __init__(self,alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val):
            self.alpha_val = alpha_val # As described in paper
            self.gamma_val = gamma_val
            self.epsilon_val = epsilon_val
            self.num_episodes = num_episodes
            self.num_iterations = num_iterations
            self.RENDERING_switch = RENDERING_switch
            pass
    
    # Creates epsilon_val greedy policy
    def epsilon_greedy_policy(self,Q, nA):
        def policy_function(obser):
            A = np.ones(nA, dtype=float) * self.epsilon_val / nA

            # Taking random action if all are same
            if np.allclose(Q[obser], Q[obser][0]):
                req_best_action = np.random.choice(
                    6, 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])[0]
            else:
                req_best_action = np.argmax(Q[obser])

            A[req_best_action] += (1.0 - self.epsilon_val)
            return A
        return policy_function

    # Q Learning algorithm implementation


    def q_learning(self,environment,alpha_val,Question,option_poilcy):
        environment.reset()
        environment.epsilon_val = self.epsilon_val
        environment.gamma_val = self.gamma_val
        environment.alpha_val = alpha_val

        # environment.render(mode='human')
        Q = defaultdict(lambda: np.zeros(environment.n_actions))
        Q[environment.terminal_state] = np.ones(environment.n_actions)

        number_of_steps = np.zeros(self.num_episodes)
        total_return = np.zeros(self.num_episodes)
        total_return_steps = np.zeros(self.num_episodes)

        for iterate in range(self.num_iterations):

            Q.clear()
            Q[environment.terminal_state] = np.ones(environment.n_actions)

            policy = self.epsilon_greedy_policy(Q, environment.n_actions)
            figcount = 0

            for i_epi in range(self.num_episodes):
                discounted_return = 0
                steps_per_epi = 0

                if (i_epi + 1) % 100 == 0:
                    print("\nIteration: {} Episode {}/{}.".format(iterate,
                                                                i_epi + 1, self.num_episodes))
                obser = environment.reset()  # Start state
                if Question == 1:
                    environment.state = 0
                    environment.start_state = 0
                else:
                    environment.state = 90
                    environment.start_state = 90

                for i in itertools.count():  # Till the end of episode
                    action_prob = policy(obser)
                    a = np.random.choice(
                        [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                    if a > 3:
                        # Passing option policy to enviroment
                        environment.options_poilcy = dill.load(
                            open(option_poilcy[a-4], 'rb'))
                        next_observation, reward, done, _ = environment.step(
                            a)  # Taking option
                    else:
                        next_observation, reward, done, _ = environment.step(
                            a)  # Taking action

                    environment.steps_count = i
                    discounted_return += reward*self.gamma_val**i  # Updating return
                    environment.discounted_return = discounted_return
                    steps_per_epi += environment.options_length

                    if self.RENDERING_switch:  # Rendering
                        environment.render(mode='human')
                        # environment.figurecount = figcount
                        # figcount += 1

                    # Finding next best action from next state
                    best_next_a = np.argmax(Q[next_observation])
                    # Q Learning update
                    Q[obser][a] += alpha_val*(reward + (self.gamma_val**environment.options_length)
                                                * Q[next_observation][best_next_a] - Q[obser][a])

                    if done:
                        print("Total discounted return is :",
                            self.gamma_val**steps_per_epi)
                        environment.discounted_return = 0
                        environment.steps_count = 0
                        break

                    obser = next_observation
                print("Total steps taken is :", steps_per_epi)
                # Updating Number of steps
                number_of_steps[i_epi] += steps_per_epi
                # Updating return
                total_return[i_epi] += discounted_return  # gamma_val**steps_per_epi
                total_return_steps[i_epi] += self.gamma_val**steps_per_epi

        number_of_steps /= self.num_iterations
        total_return /= self.num_iterations  # Updating return
        total_return_steps /= self.num_iterations

        return Q, number_of_steps, total_return, total_return_steps


    def plot_avg_steps(self,avg_steps,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, avg_steps)
        plt.title(
            "SMDP Q-Learning Average steps per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of average steps per episode")
        # plt.xlim((0, self.num_episodes))
        # plt.ylim((0, max(avg_steps)+20))
        plt.savefig(fig_save[0] + prob_type +
                    "_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        plt.xscale('log', basex=10)
        plt.savefig(fig_save[1] + prob_type +
                    "_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()


    def plot_total_return(self, total_return,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, total_return)
        plt.title(
            "SMDP Q-Learning Average Return per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        # plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[2] + prob_type +
                    "_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        # # plt.show()


    def plot_total_return_steps(self, total_return,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, total_return)
        plt.title(
            "SMDP Q-Learning Average Return per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        # plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[2] + prob_type +
                    "steps_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()

    def simulate_SMDP_Q(self,):
        # Setting some values for Q1 & Q2 where start state is centre of room 4
        option_poilcy = ["P_O1.pkl", "P_O2.pkl"]
        Question = 1
        for Question in [1, 2]:
            if Question == 1:
                mapFiles = ["map1.txt", "map2.txt"]
                prob_type = ["G1", "G2"]
                filesave = ["Q1_G1.pkl", "Q1_G2.pkl"]
                fig_title = ["Q1_G1", "Q1_G2"]
                filesave_option = [["Q1_G1_O1.pkl", "Q1_G1_O2.pkl"],
                                ["Q1_G2_O1.pkl", "Q1_G2_O2.pkl"]]
                numpy_save = ['Q_saves/Q1_avg_num_of_steps_',
                            'Q_saves/Q1_avg_return_']
                fig_save = ['Q1_Avg_steps_Prob_',
                            'Q1_Avg_steps_cut_Prob_', 'Q1_Return_Prob_']
                # option_poilcy = ["P_O1.pkl","P_O2.pkl"]

            else:
                mapFiles = ["map3.txt", "map4.txt"]
                prob_type = ["G1", "G2"]
                filesave = ["Q2_G1.pkl", "Q2_G2.pkl"]
                filesave_option = [["Q2_G1_O1.pkl", "Q2_G1_O2.pkl"],
                                ["Q2_G2_O1.pkl", "Q2_G2_O2.pkl"]]
                fig_title = ["Q2_G1", "Q2_G2"]
                numpy_save = ['Q_saves/Q2_avg_num_of_steps_',
                            'Q_saves/Q2_avg_return_']
                fig_save = ['Q2_Avg_steps_Prob_',
                            'Q2_Avg_steps_cut_Prob_', 'Q2_Return_Prob_']

          
            for pb in range(2):
                environment = gym.make("GridWorld-v0")
                environment.mapFile = mapFiles[pb]
                environment.saveFile = False
                environment.reset()
                environment.first_time = False

                Q, number_of_steps, total_return, total_return_steps = self.q_learning(environment,self.alpha_val[pb],Question,option_poilcy)

                dill.dump(Q, open(filesave[pb], 'wb'))

                np.save(numpy_save[0]+prob_type[pb], number_of_steps)
                np.save(numpy_save[1]+prob_type[pb], total_return)
                np.save(numpy_save[1]+'_steps_'+prob_type[pb], total_return_steps)

                number_of_steps = np.load(numpy_save[0]+prob_type[pb]+".npy")
                total_return = np.load(numpy_save[1]+prob_type[pb] + ".npy")
                total_return_steps = np.load(
                    numpy_save[1]+'_steps_'+prob_type[pb] + ".npy")

                self.plot_avg_steps(number_of_steps,fig_save,prob_type[pb])
                self.plot_total_return(total_return,fig_save,prob_type[pb])
                self.plot_total_return_steps(total_return_steps,fig_save,prob_type[pb])

                # Finding V values
                Q = dill.load(open(filesave[pb], 'rb'))
                environment.my_state = environment.start_state
                V = np.zeros(104)
                optimal_policy = np.zeros(104)
                optimal_policy += -1

                for k, v in Q.items():
                    if k < 104:
                        if not(np.allclose(v, v[0])):
                            optimal_policy[k] = np.argmax(v)
                        V[k] = np.max(v)

                V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
                environment.V = V
                environment.optimal_policy = optimal_policy
                environment.draw_circles = True
                environment.draw_arrows = False
                environment.figtitle = fig_title[pb]+'_'+str(self.num_episodes)
                environment.render(mode='human')

                environment.draw_circles = False
                environment.draw_arrows = True
                environment.render(mode='human')

class intra_option():

    def __init__(self,alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val):
            self.alpha_val = alpha_val # As described in paper
            self.gamma_val = gamma_val
            self.epsilon_val = epsilon_val
            self.num_episodes = num_episodes
            self.num_iterations = num_iterations
            self.RENDERING_switch = RENDERING_switch
            pass
    

    # Creates epsilon_val greedy policy
    def epsilon_greedy_policy(self,Q, nA):
        def policy_function(obser):
            A = np.ones(nA, dtype=float) * self.epsilon_val / nA

            # Taking random action if all are same
            if np.allclose(Q[obser], Q[obser][0]):
                req_best_action = np.random.choice(
                    6, 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])[0]
            else:
                req_best_action = np.argmax(Q[obser])

            A[req_best_action] += (1.0 - self.epsilon_val)
            return A
        return policy_function

    # Q Learning algorithm implementation
    def q_learning(self,environment,alpha_val,Question,option_poilcy):
        environment.reset()
        environment.epsilon_val = self.epsilon_val
        environment.gamma_val = self.gamma_val
        environment.alpha_val = alpha_val

        Q = defaultdict(lambda: np.zeros(environment.n_actions))
        Q[environment.terminal_state] = np.ones(environment.n_actions)
        Q_hat = defaultdict(lambda: np.zeros(environment.n_actions))

        number_of_steps = np.zeros(self.num_episodes)
        total_return = np.zeros(self.num_episodes)
        total_return_steps = np.zeros(self.num_episodes)


        for iterate in range(self.num_iterations):

            Q.clear()
            Q_hat.clear()
            Q[environment.terminal_state] = np.ones(environment.n_actions)
            Q_hat[environment.terminal_state] = np.ones(environment.n_actions)

            policy = self.epsilon_greedy_policy(Q, environment.n_actions)
            figcount = 0

            for i_epi in range(self.num_episodes):
                discounted_return = 0
                steps_per_epi = 0

                if (i_epi + 1) % 100 == 0:
                    print("\nIteration: {} Episode {}/{}.".format(iterate,
                                                            i_epi + 1, self.num_episodes))
                obser = environment.reset()  # Start state
                if Question == 3: # Setting start state according to Question
                    environment.state = 0
                    environment.start_state = 0
                else:
                    environment.state = 90
                    environment.start_state = 90

                for i in itertools.count():  # Till the end of episode
                    action_prob = policy(obser)
                    a = np.random.choice([i for i in range(len(action_prob))], p=action_prob)  # Action selection

                    environment.options_Q = Q # passing Q values to environment
                    environment.options_Q_hat = Q_hat  # passing Q_hat values to environment
                    if a > 3:
                        # passing option optimal policy to environment
                        environment.options_poilcy = dill.load(open(option_poilcy[a-4], 'rb'))
                    next_observation, reward, done, _ = environment.step(a)  # Taking option and also updating Q and Q_hat values accordingly
                    Q = environment.options_Q # Replacing updated Q values
                    Q_hat = environment.options_Q_hat # Replacing updated Q_hat values
                    


                    environment.steps_count = i
                    discounted_return += reward*self.gamma_val**i  # Updating return
                    environment.discounted_return = discounted_return
                    steps_per_epi += environment.options_length

                    if self.RENDERING_switch:  # Rendering
                        environment.render(mode='human')
                        # environment.figurecount = figcount
                        # figcount += 1

                    if done:
                        print("Total discounted return is :",
                            self.gamma_val**steps_per_epi)
                        environment.discounted_return = 0
                        environment.steps_count = 0
                        break

                    obser = next_observation
                print("Total steps taken is :", steps_per_epi)
                # Updating Number of steps
                number_of_steps[i_epi] += steps_per_epi
                # Updating return
                total_return[i_epi] += discounted_return
                total_return_steps[i_epi] += self.gamma_val**steps_per_epi

        number_of_steps /= self.num_iterations
        total_return /= self.num_iterations  # Updating return
        total_return_steps /= self.num_iterations

        return Q, number_of_steps, total_return, total_return_steps


    def plot_avg_steps(self,avg_steps,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, avg_steps)
        plt.title(
            "Intra option Q learning Average steps per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of average steps per episode")
        # plt.xlim((0, self.num_episodes))
        plt.ylim((0, max(avg_steps)+20))
        plt.savefig(fig_save[0] + prob_type +"_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        plt.xscale('log', basex=10)
        plt.savefig(fig_save[1] + prob_type + "_episode_" +str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()

    def plot_total_return(self,total_return,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, total_return)
        plt.title(
            "Intra option Q learning Average Return per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[2] + prob_type + "_episodes_" +str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()


    def plot_total_return_steps(self,total_return,fig_save,prob_type):
        x = np.arange(self.num_episodes)
        plt.clf()
        plt.plot(x, total_return)
        plt.title(
            "Intra option Q learning Average Return per episode for goal " + prob_type)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.xlim((0, self.num_episodes))
        plt.savefig(fig_save[2] + prob_type + "steps_episode_" + str(self.num_episodes)+'.png', dpi=1000)
        # plt.show()

    def simulate_intra_option(self):
        # Setting some values for Q3 & Q4 where start state is centre of room 4
        option_poilcy = ["P_O1.pkl", "P_O2.pkl"]
        Question = 3
        for Question in [3, 4]:

            if Question == 3:
                mapFiles = ["map1.txt", "map2.txt"]
                prob_type = ["G1", "G2"]
                filesave = ["Q3_G1.pkl", "Q3_G2.pkl"]
                fig_title = ["Q3_G1", "Q3_G2"]
                filesave_option = [["Q3_G1_O1.pkl", "Q3_G1_O2.pkl"],
                                ["Q3_G2_O1.pkl", "Q3_G2_O2.pkl"]]
                numpy_save = ['Q_saves/Q3_avg_num_of_steps_', 'Q_saves/Q3_avg_rewards_']
                fig_save = ['Q3_Avg_steps_Prob_',
                            'Q3_Avg_steps_cut_Prob_', 'Q3_Return_Prob_']

            else:
                mapFiles = ["map3.txt", "map4.txt"]
                prob_type = ["G1", "G2"]
                filesave = ["Q4_G1.pkl", "Q4_G2.pkl"]
                filesave_option = [["Q4_G1_O1.pkl", "Q4_G1_O2.pkl"],
                                ["Q4_G2_O1.pkl", "Q4_G2_O2.pkl"]]
                fig_title = ["Q4_G1", "Q4_G2"]
                numpy_save = ['Q_saves/Q4_avg_num_of_steps_', 'Q_saves/Q4_avg_rewards_']
                fig_save = ['Q4_Avg_steps_Prob_',
                            'Q4_Avg_steps_cut_Prob_', 'Q4_Return_Prob_']

            lines_circles = 1
            plot_graphs = 1


            for pb in range(2):
                environment = gym.make("GridWorld-v0")
                environment.mapFile = mapFiles[pb]
                environment.saveFile = False
                environment.reset()
                environment.first_time = False
                # environment.draw_circles = True
                # environment.figtitle = fig_title[pb]
                environment.intra_options = True

                Q, number_of_steps, total_return, total_return_steps = self.q_learning(environment,self.alpha_val[pb],Question,option_poilcy)

                dill.dump(Q, open(filesave[pb], 'wb'))

                if plot_graphs:
                    np.save(numpy_save[0]+prob_type[pb], number_of_steps)
                    np.save(numpy_save[1]+prob_type[pb], total_return)
                    np.save(numpy_save[1]+'_steps_'+prob_type[pb], total_return_steps)

                    number_of_steps = np.load(numpy_save[0]+prob_type[pb]+".npy")
                    total_return = np.load(numpy_save[1]+prob_type[pb] + ".npy")
                    total_return_steps = np.load(
                        numpy_save[1]+'_steps_'+prob_type[pb] + ".npy")

                    self.plot_avg_steps(number_of_steps,fig_save,prob_type[pb])
                    self.plot_total_return(total_return,fig_save,prob_type[pb])
                    self.plot_total_return_steps(total_return_steps,fig_save,prob_type[pb])


                # # Finding V values
                Q = dill.load(open(filesave[pb], 'rb'))
                if lines_circles:
                    environment.my_state = environment.start_state
                    V = np.zeros(104)
                    optimal_policy = np.zeros(104)
                    optimal_policy += -1

                    for k,v in Q.items():
                        if k < 104:
                            if not(np.allclose(v, v[0])):
                                optimal_policy[k] = np.argmax(v)
                            V[k] = np.max(v)
                            

                    V = preprocessing.minmax_scale(V, feature_range=(0, 0.5)) 
                    environment.V = V # passing V values to environment 
                    environment.optimal_policy = optimal_policy # passing optimal policy to environment
                    environment.draw_circles = True
                    environment.draw_arrows = False
                    environment.figtitle = fig_title[pb]+'_'+str(self.num_episodes)
                    environment.render(mode='human')

                    environment.draw_circles = False
                    environment.draw_arrows = True
                    environment.render(mode='human')

class plotting():

    def __init__(self):
        pass
    
    def plot_avg_steps(self,num_episodes, avg_steps_loads, avg_steps_labels,savefile,goal):
        x = np.arange(num_episodes)
        plt.clf()
        for i in range(len(avg_steps_loads)):
            avg_steps = np.load(avg_steps_loads[i])
            plt.plot(x, avg_steps, label=avg_steps_labels[i])
        plt.title(
            "Average steps per episode for goal G"+goal)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of average steps per episode")
        plt.legend(loc=0)
        plt.xscale('log', basex=10)
        plt.savefig(savefile+'.png', dpi=1000)
        # plt.show()


    def plot_total_return(self,num_episodes, total_return_loads, labels, savefile, goal):
        x = np.arange(num_episodes)
        plt.clf()
        for i in range(len(total_return_loads)):
            total_return = np.load(total_return_loads[i])
            plt.plot(x, total_return, label=labels[i])

        plt.title(
            "Average Total return per episode for goal G" + goal)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.legend(loc=0)
        # plt.xlim((0, self.num_episodes))
        plt.xscale('log', basex=10)
        plt.savefig(savefile+'.png', dpi=1000)
        # # plt.show()

    def generate_plots(self):

        num_episodes = 10000
        avg_steps_labels = ['SMDP: Room 1 Top left', 'SMDP: Room 4 Center',
                            'Intra-Option: Room 1 Top left', 'Intra-Option: Room 4 center']
        avg_steps_loads_G1 = ['Q_saves/Q1_avg_num_of_steps_G1.npy', 'Q_saves/Q2_avg_num_of_steps_G1.npy',
                        'Q_saves/Q3_avg_num_of_steps_G1.npy', 'Q_saves/Q4_avg_num_of_steps_G1.npy']
        avg_steps_loads_G2 = ['Q_saves/Q1_avg_num_of_steps_G2.npy', 'Q_saves/Q2_avg_num_of_steps_G2.npy',
                        'Q_saves/Q3_avg_num_of_steps_G2.npy', 'Q_saves/Q4_avg_num_of_steps_G2.npy']


        avg_return_load_G1 = ['Q_saves/Q1_avg_return_G1.npy', 'Q_saves/Q2_avg_return_G1.npy',
                            'Q_saves/Q3_avg_rewards_G1.npy', 'Q_saves/Q4_avg_rewards_G1.npy']
        avg_return_load_G2 = ['Q_saves/Q1_avg_return_G2.npy', 'Q_saves/Q2_avg_return_G2.npy',
                            'Q_saves/Q3_avg_rewards_G2.npy', 'Q_saves/Q4_avg_rewards_G2.npy']

        avg_return_steps_load_G1 = ['Q_saves/Q1_avg_return__steps_G1.npy', 'Q_saves/Q2_avg_return__steps_G1.npy',
                                    'Q_saves/Q3_avg_rewards__steps_G1.npy', 'Q_saves/Q4_avg_rewards__steps_G1.npy']
        avg_return_steps_load_G2 = ['Q_saves/Q1_avg_return__steps_G2.npy', 'Q_saves/Q2_avg_return__steps_G2.npy',
                                    'Q_saves/Q3_avg_rewards__steps_G2.npy', 'Q_saves/Q4_avg_rewards__steps_G2.npy']

        # avg_steps_labels[1], avg_steps_labels[2] = avg_steps_labels[2], avg_steps_labels[1]
        # avg_steps_loads_G1[1], avg_steps_loads_G1[2] = avg_steps_loads_G1[2], avg_steps_loads_G1[1]
        # avg_steps_loads_G2[1], avg_steps_loads_G2[2] = avg_steps_loads_G2[2], avg_steps_loads_G2[1]
        # avg_return_steps_load_G1[1], avg_return_steps_load_G1[2] = avg_return_steps_load_G1[2], avg_return_steps_load_G1[1]
        # avg_return_steps_load_G2[1], avg_return_steps_load_G2[2] = avg_return_steps_load_G2[2], avg_return_steps_load_G2[1]


        self.plot_avg_steps(num_episodes, avg_steps_loads_G1, avg_steps_labels,'Avg_steps_G1','1')
        self.plot_avg_steps(num_episodes, avg_steps_loads_G2,
                    avg_steps_labels, 'Avg_steps_G2','2')

        self.plot_total_return(num_episodes, avg_return_load_G1,
                        avg_steps_labels, 'Total_return_G1', '1')
        self.plot_total_return(num_episodes, avg_return_load_G2,
                        avg_steps_labels, 'Total_return_G2', '2')


        self.plot_total_return(num_episodes, avg_return_steps_load_G1,
                                avg_steps_labels, 'Total_return_steps_G1', '1')
        self.plot_total_return(num_episodes, avg_return_steps_load_G2,
                                avg_steps_labels, 'Total_return_steps_G2', '2')


if __name__ == "__main__":

    alpha_val = [0.12, 0.25] # As described in paper

    num_episodes = 10000
    num_iterations = 1
    RENDERING_switch = False
    gamma_val=0.9
    epsilon_val=0.3
    if(len(sys.argv) <= 1 ):
        print('len(sys.argv): ', len(sys.argv))
        print('Please pass the argument ')
        exit()

    if(sys.argv[1] == 'learn_policy'):
        from Q1 import learn_option_policy as LOP
        lop = LOP(alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val)
        lop.generate_option_policy()
    
    if(sys.argv[1] == 'smdp'):
        from Q1 import SMDP_Q as SMDP
        smdp = SMDP(alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val)
        smdp.simulate_SMDP_Q()
        
    
    if(sys.argv[1] == 'intra'):
        from Q1 import intra_option as Intra
        intra = Intra(alpha_val,num_episodes,num_iterations,RENDERING_switch,gamma_val,epsilon_val)
        intra.simulate_intra_option()

    if(sys.argv[1] == 'plots'):
        from Q1 import plotting 
        plots = plotting()
        plots.generate_plots()
        