# Taken some reference & idea from below github soucre: Some GridWorld envirironments for OpenAI Gym
# https://github.com/opocaj92/GridWorldenvirs 

import numpy as np
import gym
import gridworld
import itertools
from collections import defaultdict
import sys
from gym import wrappers
import dill
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

pklfiles = ['PuddleA_Q.pkl', 'PuddleB_Q.pkl', 'PuddleC_Q.pkl']
maps = ['PuddleWorldA.txt', 'PuddleWorldB.txt', 'PuddleWorldC.txt']
fig_title = ['Puddle World A','Puddle World B', 'Puddle World C']
plotsave = ['AvgSteps_A', 'AvgSteps_B', 'AvgSteps_B']
select_GridWorldMap = ['A','B','C']
movie_names = ['movieA', 'movieB', 'movieC']
lambda_values = [0,0.2,0.5,0.9,0.99,1]


#########################################
###########   Plotting Class   ##########
#########################################
# For plotting all three problems togeather
class GeneratePlots():

    def __init__(self,dir_name):
        self.dir_name = dir_name
        pass
        
    def plot_avg_steps(self):
        '''
        These method will plot average steps taken in order to learn
        It will plot all three methods combinely 
        '''
        plt.clf()
        plot_a = np.load(self.dir_name+'/avg_num_of_steps_for_problem_' +select_GridWorldMap[0]+'.npy')
        plot_b = np.load(self.dir_name+'/avg_num_of_steps_for_problem_' +select_GridWorldMap[1]+'.npy')
        plot_c = np.load(self.dir_name+'/avg_num_of_steps_for_problem_' +select_GridWorldMap[2]+'.npy')
        episodes_count_a = len(plot_a)
        episodes_count_b = len(plot_b)
        episodes_count_c = len(plot_c)
        plt.plot(np.arange(episodes_count_a), plot_a)
        plt.plot(np.arange(episodes_count_a), plot_b)
        plt.plot(np.arange(episodes_count_a), plot_c)
        plt.legend(['Goal A','Goal B','Goal C'],loc=0)
        plt.title('Average number of steps taken to reach goal')
        plt.xlabel('Number of episodes')
        plt.ylabel('Average number of steps')
        plt.savefig('combine_plot_of_avg_steps.png', dpi=300)
        # plt.show()


    def plot_total_return(self):
        plt.clf()
        plot_a = np.load(self.dir_name+'/total_return_per_episode_for_problem_' + select_GridWorldMap[0]+'.npy')
        plot_b = np.load(self.dir_name+'/total_return_per_episode_for_problem_' + select_GridWorldMap[1]+'.npy')
        plot_c = np.load(self.dir_name+'/total_return_per_episode_for_problem_' + select_GridWorldMap[2]+'.npy')
        episodes_count_a = len(plot_a)        
        episodes_count_b = len(plot_b)
        episodes_count_c = len(plot_c)
        plt.plot(np.arange(episodes_count_a), plot_a)
        plt.plot(np.arange(episodes_count_a), plot_b)
        plt.plot(np.arange(episodes_count_a), plot_c)
        plt.legend(['Goal A', 'Goal B', 'Goal C'], loc=0)
        plt.title('Average return per episode')
        plt.xlabel('Number of episodes')
        plt.ylabel('Average Return')
        plt.savefig('combine_plot_of_avg_return.png', dpi=300)

    def lambda_avg_step(self,GridWorldMap,slambda_val):
        '''
        This method will plot avg step for specific world for all lambda values 
        '''
    
        for i in range(len(slambda_val)):
            load_data = np.load(self.dir_name+'/avg_num_of_steps_for_problem_' + GridWorldMap+'_lambda_'+str(slambda_val[i])+'.npy')
            episodes_count= len(load_data)        
            plt.plot(np.arange(episodes_count),load_data)
        plt.legend(['Lambda: 0', 'Lambda: 0.3', 'Lambda: 0.5','Lambda: 0.9', 'Lambda: 0.99', 'Lambda: 1'], loc=0)
        plt.title('Sarsa lambda Average number of steps to goal fro problem'+ str(GridWorldMap))
        plt.xlabel('Number of episodes')
        plt.ylabel('Average steps ')
        plt.xlim(0, episodes_count)
        plt.savefig('Combined_Avg_steps_'+str(GridWorldMap)+'_'+str(num_episodes)+'.png', dpi=300)
    
    def lambda_total_return(self,GridWorldMap,slambda_val):
        '''
        This method will plot average return for specific world for all lambda values 
        '''
        plt.clf()
        for i in range(len(slambda_val)):
            load_data = np.load(self.dir_name+'/total_return_per_episode_for_problem_' + str(GridWorldMap) +'_lambda_'+str(slambda_val[i])+'.npy')
            episodes_count= len(load_data)        
            plt.plot(np.arange(episodes_count),load_data)
            num_episodes = len(load_data)
            print('num_episodes: ', num_episodes)
        plt.legend(['Lambda: 0', 'Lambda: 0.3', 'Lambda: 0.5','Lambda: 0.9', 'Lambda: 0.99', 'Lambda: 1'], loc=0)
        plt.title('Sarsa lambda Average return per episode for problem '+ str(GridWorldMap))
        plt.xlabel('Number of episodes')
        plt.ylabel('Average Return')
        plt.xlim(0, episodes_count)
        plt.savefig('Combined_return_'+str(GridWorldMap)+'_'+str(num_episodes)+'.png', dpi=300)
        # plt.show()

    # For plotting all three problems togeather
    def lambda_plots(self,select_GridWorldMap,lambda_values):
        plt.clf()

        for i in range(len(select_GridWorldMap)):
            self.lambda_avg_step(select_GridWorldMap[i],lambda_values)
            self.lambda_total_return(select_GridWorldMap[i],lambda_values)
            plt.clf()
       # plt.show()


###################
# General methods #
###################

def GenerateMovie(moviefilename,fps=5,learning_method='ANY'):
    '''
    method wil create movies and removes unnecessary files
    This method requires ffmepg library 
    '''
    os.system("rm "+str(learning_method)+'_'+str(moviefilename)+".mp4")
    os.system("ffmpeg -r "+str(fps) +" -i _tmp%05d.png  "+str(learning_method)+str(moviefilename)+".mp4")
    os.system("rm _tmp*.png")


# Creates epsilon greedy policy
def eps_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action_picked = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best_action_picked = np.argmax(Q[observation])

        A[best_action_picked] += (1.0 - epsilon)
        return A
    return policy_fn

# find optimal policy
def find_optimal_policy(Q):
    opt_policy = defaultdict(lambda: np.zeros(1))
    for k, v in Q.items():
        if np.allclose(v, v[0]):
            # np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            opt_policy[k] = 1
        else:
            opt_policy[k] = np.argmax(v)
    return opt_policy


def plot_avg_steps(num_episodes,avg_steps,selected_GridWorldMap):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, avg_steps)
    plt.title("Average number of steps to reach goal for problem " + str(selected_GridWorldMap))
    plt.xlabel("Episodes")
    plt.ylabel("Average number of steps")
    plt.xlim((0, num_episodes))
    plt.savefig("Avg_steps_"+str(selected_GridWorldMap)+str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return(num_episodes,total_return,selected_GridWorldMap):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, total_return)
    plt.title("Return per episode for problem " + str(selected_GridWorldMap))
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.xlim((0, num_episodes))
    plt.savefig("Return_"+str(selected_GridWorldMap)+ str(num_episodes)+'.png', dpi=300)

######################################
# Q Learning algorithm implementation#
######################################
def QL_ALGO(envir,selected_GridWorldMap,learning_method, num_episodes=500, iterat=50, gamma=0.9, alpha=0.1, epsilon=0.1,RENDERING=False):
    envir.reset()

    # envir.render(mode='human')
    Q = defaultdict(lambda: np.zeros(envir.n_actions))

    for itr in range(iterat):
        number_of_steps = np.zeros(num_episodes)
        total_return = np.zeros(num_episodes)

        Q.clear()

        policy = eps_greedy_policy(Q, epsilon, envir.n_actions)
        figure_counter = 0

        for i_eps in range(num_episodes):
            discounted_return = 0

            if (i_eps + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,i_eps + 1, num_episodes))
            observation = envir.reset()  # Start state

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                action_picked = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                next_observation, reward, done, _ = envir.step(action_picked)  # Taking action

                envir.steps = i
                discounted_return += reward  # Updating return
                envir.discounted_return = discounted_return

                if RENDERING:  # Rendering
                    envir.render(mode='human')
                    envir.figurecount = figure_counter
                    figure_counter += 1

                # Finding next best action from next state
                best_next_action = np.argmax(Q[next_observation])
                Q[observation][action_picked] += alpha *(reward + gamma*Q[next_observation][best_next_action] - Q[observation][action_picked])  # Q Learning update

                if done:
                    envir.discounted_return = 0
                    envir.steps = 0
                    break

                observation = next_observation
            number_of_steps[i_eps] = i  # Updating Number of steps
            total_return[i_eps] = discounted_return  # Updating return


        np.save(learning_method+'/avg_num_of_steps_for_problem_' +selected_GridWorldMap+"_itr_"+str(itr), number_of_steps)
        np.save(learning_method+'/total_return_per_episode_for_problem_' +selected_GridWorldMap+"_itr_"+str(itr), total_return)

    return Q

#####################################
# Simulate on learned optimal policy#
#####################################

def QL_SIMULATION(envir,opt_policy, num_episodes=10,RECORDING=False,RENDERING=False):
    figure_counter = 0
    envir.saveFile = RECORDING

    for i in range(num_episodes):
        discounted_return = 0
        observation = envir.reset()

        for i in itertools.count():

            action_picked = int(opt_policy[observation])
            next_observation, reward, done, _ = envir.step(action_picked)

            envir.steps = i
            discounted_return += reward
            envir.discounted_return = discounted_return

            if RENDERING:
                envir.render(mode='human')
                envir.figurecount = figure_counter
                figure_counter += 1

            if done:
                envir.discounted_return = 0
                envir.steps = 0
                break

            observation = next_observation
    

#####################################
# Simulate on learned optimal policy#
#####################################
def SARSA_SIMULATION(envir,opt_policy, num_episodes=10,RECORDING=False,RENDERING=False):
    figure_counter = 0
    envir.saveFile = RECORDING

    for i in range(num_episodes):
        discounted_return = 0
        observation = envir.reset()

        for i in itertools.count():

            action_picked = int(opt_policy[observation])
            next_observation, reward, done, _ = envir.step(action_picked)

            envir.steps = i
            discounted_return += reward
            envir.discounted_return = discounted_return

            if RENDERING:
                envir.render(mode='human')
                envir.figurecount = figure_counter
                figure_counter += 1

            if done:
                envir.discounted_return = 0
                envir.steps = 0
                break

            observation = next_observation


########################################
# Sarsa lambda algorithm implementation#
########################################

def SARSA_LAMBDA_ALGO(envir, selected_world, learning_method, num_episodes=1000, iterat=50, gamma=0.9, slambda_val=0.1, alpha=0.1, epsilon=0.1,RENDERING=False):
    print('selected_world: ', selected_world)
    envir.reset()
    Q = defaultdict(lambda: np.zeros(envir.n_actions))        
    et = defaultdict(lambda: np.zeros(envir.n_actions))


    for itr in range(iterat):
        number_of_steps = np.zeros(num_episodes)
        total_return = np.zeros(num_episodes)        
        Q.clear()
        et.clear()

        policy = eps_greedy_policy(Q, epsilon, envir.n_actions)
        figure_counter = 0

        for i_eps in range(num_episodes):
            discounted_return = 0

            if (i_eps + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,i_eps + 1, num_episodes))

            # Reset the envirironment and pick the first action
            observation = envir.reset()
            action_prob = policy(observation)
            action_picked = np.random.choice([i for i in range(len(action_prob))], p=action_prob)

            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                next_observation, reward, done, _ = envir.step(action_picked)

                envir.steps = i
                discounted_return += reward  # Updating return
                envir.discounted_return = discounted_return

                if RENDERING:  # Rendering
                    envir.render(mode='human')
                    envir.figurecount = figure_counter
                    figure_counter += 1

                
                next_action_probs = policy(next_observation)
                next_action = np.random.choice(
                    np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                TDError = reward + gamma * \
                    Q[next_observation][next_action] - Q[observation][action_picked] # TD Error
                et[observation][action_picked] += 1  # Accumulating traces

                for k, _ in Q.items():
                    for actions in range(4):
                        Q[k][actions] += alpha*TDError*et[k][actions] # Sarsa lambda Q update
                        et[k][actions] = gamma*slambda_val*et[k][actions] # Elegibility trace update

                if done:
                    # print("Total discounted return is :", discounted_return)
                    envir.discounted_return = 0
                    envir.steps = 0
                    break

                observation = next_observation
                action_picked = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_eps] = i  # Updating Number of steps
            total_return[i_eps] = discounted_return  # Updating return

        np.save(learning_method+'/avg_num_of_steps_for_problem_' +selected_world+"_lambda_"+str(slambda_val)+"_itr_"+str(itr), number_of_steps)
        np.save(learning_method+'/total_return_per_episode_for_problem_' +selected_world+"_lambda_"+str(slambda_val)+"_itr_"+str(itr), total_return)
    return Q

#################################
# Sarsa algorithm implementation#
#################################
def SARSA_ALGO(envir, selected_world,learning_method, num_episodes=1000, iterat=50, gamma=0.9, alpha=0.1, epsilon=0.2,RENDERING=False):
    envir.reset()
    Q = defaultdict(lambda: np.zeros(envir.n_actions))

    for itr in range(iterat):
        number_of_steps = np.zeros(num_episodes)
        total_return = np.zeros(num_episodes)

        Q.clear()

        policy = eps_greedy_policy(Q, epsilon, envir.n_actions)
        figure_counter = 0

        for i_eps in range(num_episodes):
            discounted_return = 0

            if (i_eps + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,
                                                                i_eps + 1, num_episodes))
    
            # Reset the envirironment and pick the first action
            observation = envir.reset()
            action_prob = policy(observation)
            action_picked = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)

            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                next_observation, reward, done, _ = envir.step(action_picked)

                envir.steps = i
                discounted_return += reward  # Updating return
                envir.discounted_return = discounted_return

                if RENDERING: # Rendering
                    envir.render(mode='human')
                    envir.figurecount = figure_counter
                    figure_counter += 1

                next_action_probs = policy(next_observation)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                Q[observation][action_picked] += alpha * \
                    (reward + gamma*Q[next_observation]
                        [next_action] - Q[observation][action_picked]) # Sarsa update

                if done:
                    # print("Total discounted return is :", discounted_return)
                    envir.discounted_return = 0
                    envir.steps = 0
                    break

                observation = next_observation
                action_picked = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_eps] = i  # Updating Number of steps
            total_return[i_eps] = discounted_return  # Updating return

        np.save(learning_method+'/avg_num_of_steps_for_problem_' + selected_world+"_itr_"+str(itr), number_of_steps)
        np.save(learning_method+'/total_return_per_episode_for_problem_' + selected_world+"_itr_"+str(itr), total_return)
    return Q

def simulate(learning_method,num_episodes,iterat):
    '''
    This method will run the code based on the type learning technique passed 
    '''

    for selected_world in range(len(select_GridWorldMap)):

        envir = gym.make("GridWorld-v0")
        filename = pklfiles[selected_world]
        RECORDING = False
        RENDERING = False
        envir.saveFile = RECORDING
        envir.mapFile = maps[selected_world]
        envir.figtitle = fig_title[selected_world]

        if select_GridWorldMap[selected_world] == "C":
            envir.westerly_wind = False

        envir.reset()
        envir.first_time = False


        if(learning_method == 'SARSA_LAMBDA'):    
            for l in range(len(lambda_values)):
                print('Working with SARSA_LAMBDA')
                envir.figtitle = fig_title[selected_world]+" lambda "+str(lambda_values[l])
                print('select_GridWorldMap[selected_world]: ', select_GridWorldMap[selected_world])
                Q = SARSA_LAMBDA_ALGO(envir,select_GridWorldMap[selected_world],learning_method, num_episodes=num_episodes, iterat=iterat,gamma=0.9, slambda_val=lambda_values[l] ,alpha=0.1, epsilon=0.4,RENDERING=RENDERING)

                if RECORDING:
                    GenerateMovie(movie_names[selected_world], 5,select_GridWorldMap[selected_world])

                # Plotting
                avg_number_of_steps = np.zeros(num_episodes)
                total_return_per_episode = np.zeros(num_episodes)

                for i in range(iterat):
                    avg_number_of_steps += np.load(learning_method+'/avg_num_of_steps_for_problem_' + select_GridWorldMap[selected_world]+"_lambda_"+str(lambda_values[l])+"_itr_"+str(i)+".npy")
                    total_return_per_episode += np.load(learning_method+'/total_return_per_episode_for_problem_' + select_GridWorldMap[selected_world]+"_lambda_"+str(lambda_values[l])+"_itr_"+str(i)+".npy")

                avg_number_of_steps /= iterat
                total_return_per_episode /= iterat
                np.save(learning_method+'/avg_num_of_steps_for_problem_' + select_GridWorldMap[selected_world]+"_lambda_"+str(lambda_values[l]), avg_number_of_steps)
                np.save(learning_method+'/total_return_per_episode_for_problem_' +select_GridWorldMap[selected_world]+"_lambda_"+str(lambda_values[l]), total_return_per_episode)

                # Finding optimal policy based on Q
                opt_policy = find_optimal_policy(Q)
                envir.opt_policy = opt_policy

                dill.dump_session(filename)
                
                # envir.saveFile = True
                # Drawing arrows of optimal policy
                envir.draw_arrows = True
                envir.render(mode='human')

                # For recording video of simulation
                # envir.draw_arrows = False
                RENDERING = False
                RECORDING = False

                if RECORDING:
                    SARSA_SIMULATION(envir,opt_policy, 10,RECORDING,RENDERING)            
                    # GenerateMovie(movie_names[selected_world], 5,select_GridWorldMap[selected_world])
            # plot_avg_steps(num_episodes, avg_number_of_steps,select_GridWorldMap[selected_world])
            # plot_total_return(num_episodes, total_return_per_episode,select_GridWorldMap[selected_world])

        else:
            if(learning_method == 'QL'):
                Q = QL_ALGO(envir,select_GridWorldMap[selected_world],learning_method, num_episodes=num_episodes, iterat=iterat,gamma=0.9, alpha=0.1, epsilon=0.3,RENDERING=RENDERING)
            if(learning_method == 'SARSA'):      
                Q = SARSA_ALGO(envir,select_GridWorldMap[selected_world],learning_method, num_episodes=num_episodes, iterat=iterat,gamma=0.9, alpha=0.1, epsilon=0.3,RENDERING=RENDERING)

            if RECORDING:
                GenerateMovie(movie_names[selected_world], 5,select_GridWorldMap[selected_world])


            # Plotting
            avg_number_of_steps = np.zeros(num_episodes)
            total_return_per_episode = np.zeros(num_episodes)

            for i in range(iterat):
                avg_number_of_steps += np.load(learning_method+'/avg_num_of_steps_for_problem_' + select_GridWorldMap[selected_world]+"_itr_"+str(i)+".npy")
                total_return_per_episode += np.load(learning_method+'/total_return_per_episode_for_problem_' + select_GridWorldMap[selected_world]+"_itr_"+str(i)+".npy")

            avg_number_of_steps /= iterat
            total_return_per_episode /= iterat
            np.save(learning_method+'/avg_num_of_steps_for_problem_' + select_GridWorldMap[selected_world], avg_number_of_steps)
            np.save(learning_method+'/total_return_per_episode_for_problem_' + select_GridWorldMap[selected_world], total_return_per_episode)

            plot_avg_steps(num_episodes, avg_number_of_steps,select_GridWorldMap[selected_world])
            plot_total_return(num_episodes, total_return_per_episode,select_GridWorldMap[selected_world])

            # Finding optimal policy based on Q
            opt_policy = find_optimal_policy(Q)
            envir.opt_policy = opt_policy

            dill.dump_session(filename)
            
            # envir.saveFile = True
            # Drawing arrows of optimal policy
            envir.draw_arrows = True
            envir.render(mode='human')

            # For recording video of simulation
            # envir.draw_arrows = False
            RENDERING = True
            RECORDING = True

            if RECORDING:
                if(learning_method == 'QL'):
                    QL_SIMULATION(envir,opt_policy, 10,RECORDING,RENDERING)
                if(learning_method == 'SARSA'):
                    SARSA_SIMULATION(envir,opt_policy, 10,RECORDING,RENDERING)   
                         
                GenerateMovie(movie_names[selected_world], 5,select_GridWorldMap[selected_world])

    # Plotting all 3 problems graph togeather
    if(learning_method == 'QL' or learning_method == 'SARSA'):
        plotting = GeneratePlots(learning_method)
        plotting.plot_avg_steps()
        plotting.plot_total_return()
    if(learning_method == 'SARSA_LAMBDA'):
        plotting = GeneratePlots(learning_method)
        plotting.lambda_plots(select_GridWorldMap,lambda_values)


if __name__ == '__main__':
    import sys
    arguments_len = len(sys.argv)
    if(arguments_len < 2):
        print('Please pass the argument available option: -s QL, SARSA, SARSA_LAMBDA')
    elif(arguments_len < 2):
        print('Please pass only required arguments and check README')
    else:
        if(sys.argv[2] in {'QL', 'SARSA', 'SARSA_LAMBDA'}):    
            learning_method = sys.argv[2]
            print('Learning_method: ', learning_method)
            num_episodes = 500
            iterat = 50
            simulate(learning_method,num_episodes,iterat)
        else:
            print('Please pass the argument available option: -s QL, SARSA, SARSA_LAMBDA')

