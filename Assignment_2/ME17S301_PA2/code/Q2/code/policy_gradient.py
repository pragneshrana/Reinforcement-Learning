#!/usr/bin/env python

# import click
# click.disable_unicode_literals_warning = True
import numpy as np
import gym
import itertools
import dill
from gym import wrappers
from rlpa2 import chakra
from rlpa2 import vishamC

class policy_gradient():

    def __init__(self):
        pass

    def bias_inclusion(self,curr_observation):
        '''
        method includes bias in the observation
        '''
        return [curr_observation[0], curr_observation[1], 1]

    def chakra_get_action(self,theta, curr_observation, distribution=np.random):
        ob_1 = self.bias_inclusion(curr_observation)
        cal_mean = theta.dot(ob_1)
        return distribution.normal(loc=cal_mean, scale=1.)


    def get_cal_mean(self,theta, curr_observation):
        ob_1 = self.bias_inclusion(curr_observation)
        cal_mean = theta.dot(ob_1)
        return cal_mean

    def calculate_logPi(self,action, theta, curr_observation):
        ob_1 = self.bias_inclusion(curr_observation)
        cal_mean = theta.dot(ob_1)
        diff = action - cal_mean
        return -0.5 * np.log(2 * np.pi) * theta.shape[0] - 0.5 * np.sum(np.square(diff))

    def calculate_grad_logPi(self,action, theta, state):      
        state_1 = self.bias_inclusion(state)
        cal_mean = theta.dot(state_1)
        diff = np.array(action - cal_mean)
        grad = np.outer(diff, state_1)
        return grad

    def train_environment(self,env_id):
        # Register the environment
        distribution = np.random.RandomState(42)

        if(env_id == 'chakra'):
            from rlpa2 import chakra
            env = gym.make('chakra-v0')
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
        elif(env_id == 'vishamC'):
            from rlpa2 import vishamC
            env = gym.make('vishamC-v0')
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError('Unsupported environment')

        env.seed(42)
        timestep_limit = env.spec.max_episode_steps

        # alpha = 0.5
        # gamma = 0.9
        # batch_size = 100

        batch_size_list = [100]
        alpha_list = [0.9]
        gamma_list = [0.9]
        iteration = 100

        for batch_size in batch_size_list:
            for alpha in alpha_list:
                for gamma in gamma_list:
                    # Initialize parameters
                    theta = distribution.normal(scale=0.01, size=(action_dim, obs_dim + 1))

                    # Store baselines for each time step.
                    baselines = np.zeros(timestep_limit)
                    episode_len = 0
                    avg_episode_rewards = []

                    # Policy Implementation
                    for itr in range(iteration):
                        # n_samples = 0
                        grad = np.zeros_like(theta)
                        episode_rewards = []

                        # Storing cumulative returns for each time step
                        all_returns_for_baseline = [[] for _ in range(timestep_limit)]
                    
                        # Iterating over Batch Size
                        for current_batch in range(batch_size): 
                            
                            # Collect a new trajectory
                            rewards_obtained = []
                            states = []
                            actions = []
                            curr_observation = env.reset()

                            for i in itertools.count():
                                action = self.chakra_get_action(theta, curr_observation, distribution=distribution)
                                try:
                                    next_obsevation, reward, reached, _ = env.step(action)
                                except TypeError: 
                                    env.reset()
                                states.append(curr_observation)
                                actions.append(action)
                                rewards_obtained.append(reward)
                                curr_observation = next_obsevation
                                # if(itr >= iteration -2:
                                    # env.render()
                                if(reached):
                                    # print('\n reached and reward is', reward)
                                    episode_len = i
                                    env.done = False
                                    break

                            # print('Episode reward: %.2f Length: %d' %(np.sum(rewards_obtained), episode_len))

                            # Going back in time to compute returns and Compute accumulate gradient
                            for t in range(episode_len):
                                tt = t
                                G_t = 0
                                while tt < episode_len:
                                    G_t += gamma**(tt-t)*rewards_obtained[tt]
                                    tt += 1
                                all_returns_for_baseline[t].append(G_t)
                                advantages = G_t - baselines[t]
                                grad_temp = self.calculate_grad_logPi(actions[t], theta, states[t])
                                grad_temp *= advantages
                                grad += grad_temp

                            episode_rewards.append(np.sum(rewards_obtained))

                            # Updating Baseline from earlier trajectory
                            base_temp = np.zeros(len(all_returns_for_baseline))
                            for t in range(len(all_returns_for_baseline)):
                                if(len(all_returns_for_baseline[t]) > 0):
                                    base_temp[t] = np.cal_mean(all_returns_for_baseline[t])
                            
                            baselines = base_temp
                                
                        # Normalizing Gradient
                        grad = grad / (np.linalg.norm(grad) + 1e-8)
                        theta += alpha*grad  # /batch_size
                        avg_episode_rewards.append(np.cal_mean(episode_rewards))
                        print('\n Iteration: %d Average Return: %.2f |theta|_2: %.2f' %(itr, avg_episode_rewards[itr], np.linalg.norm(theta)))
                        print('Theta', theta)
                    
                    # dill.dump_session('chakra.pkl')
                    np.save(str(env_id),theta)
                    np.save('avg_'+str(env_id)+'_episode_rewards_lr_'+str(alpha)+'_gma_'+str(gamma)+'_bs_'+str(batch_size), avg_episode_rewards)
        
    def roll_out(self,env_id):
        # Loading trained theta
        theta = np.load(str(env_id)+'.npy')
        env = gym.make(str(env_id)+'-v0')
        distribution = np.random.RandomState(42)

        # For calculating value function
        value_fun_lst = []
        state_space = []
        for i in range(20):
            value = 0
            curr_observation = env.reset()
            state_space.append(curr_observation)
            for i in itertools.count():
                action = self.chakra_get_action(theta,curr_observation, distribution=distribution)
                next_obsevation, reward, reached, _ = env.step(action)
                value += 0.9*i*reward
                curr_observation = next_obsevation
                env.render()
                if(reached):
                    env.done = False
                    break
            value_fun_lst.append(value)
        
        print(state_space)
        print(value_fun_lst)
        np.save(str(env_id)+'_state_space ',state_space)
        np.save(str(env_id)+'_val_fun ',value_fun_lst)


        # For Saving policy trajectries
        for epi in range(20):
            curr_observation = env.reset()
            trajectory = []
            for i in itertools.count():
                trajectory.append(curr_observation)
                action = self.chakra_get_action(theta, curr_observation, distribution=distribution)
                next_obsevation, reward, reached, _ = env.step(action)
                curr_observation = next_obsevation
                env.render()
                if(reached):
                    print('\n reached and reward is', reward)
                    episode_len = i
                    env.done = False
                    np.save(str(env_id)+'_traj_'+str(epi),trajectory)
                    break


    def PlotTrajectory(self,env_id):
        '''
        Method will plot trajectory once  learning is done 
        ''' 
        #generating mesh grids
        x_values = np.linspace(-1.0, 1.0, 200)
        y_values = np.linspace(-1.0, 1.0, 200)
        X, Y = np.meshgrid(x_values, y_values)
        Z = (.5*X**2 + 5*Y**2)

        #ploting figures
        plt.figure()
        cp = plt.contour(X, Y, Z)
        plt.clabel(cp, inline=True,fontsize=10)
        for i in range(15):
            tmp = np.load(str(env_id)+'traj_'+str(i)+'.npy')
            x = []
            y = []
            for j in range(len(tmp)):
                x.append(tmp[j][0])
                y.append(tmp[j][1])
            plt.plot(x, y)
        plt.title('Trajectory of Policy after leraning for ',env_id)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('Trajectory_'+str(env_id)+'.png', dpi=300)
        # plt.show()
    
    def ValueFuncPlot(self,env_id):
        '''
        This method will calculate value function based on the saved learning agent 
        and plot the value function 
        '''
        x_values = np.linspace(-1.0, 1.0, 200)
        y_values = np.linspace(-1.0, 1.0, 200)
        X, Y = np.meshgrid(x_values, y_values)
        Z = (.5*X**2 + 5*Y**2)

        #plotting figures
        plt.figure()
        cp = plt.contour(X, Y, Z)
        plt.clabel(cp, inline=True,
                fontsize=10)
        state_space = np.load('StateSpace'+str(env_id)+'.npy')
        val = np.load('ValueFunc'+str(env_id)+'.npy')
        val = np.around(val, decimals=2)
        x = []
        y = []
        for j in range(len(state_space)):
            x.append(state_space[j][0])
            y.append(state_space[j][1])
        plt.scatter(x, y,marker='o')

        for i, txt in enumerate(val):
            plt.annotate(txt, (x[i], y[i]), ha='center', size='large')
        plt.title('Value Function for '+str(env_id)+'after learning')
        plt.xlabel('X val ')
        plt.ylabel('Y val')
        plt.savefig('Val_Func_'+str(env_id)+'.png', dpi=300)
        plt.show()
        
    def plot_avg_ret(self,env_id,batch_size_list,alpha_list,gamma_list,iteration):
        plt.clf()
        # matplotlib.rcParams['figure.figsize'] = (10, 16)
        plt.rcParams['figure.figsize'] = (15,25)
        x = np.arange(iteration)
        for alpha in alpha_list:
            for batch_size in batch_size_list:
                for gamma in gamma_list:
                    load_saved = np.load('avg_episode_rewards_lr_'+str(alpha)+'_gma_' +str(gamma)+'_bs_'+str(batch_size)+'.npy')
                    plt.plot(x, load_saved,label=' Gamma:'+str(gamma)+'alpha(learning rate):'+str(alpha)+' Gamma:'+str(gamma)+'Batch size:'+str(batch_size))
        plt.legend(loc=0,frameon=False)
        plt.title('Average return obtained by batch size')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Average Return by batch')
        plt.xlim(0, iteration)
        plt.savefig('avg_return'+str(env_id)+'.png', dpi=300)
        plt.show()


if __name__ == '__main__':

    import policy_gradient as RO
    from rlpa2 import chakra

    #### callling function 
    import sys
    arguments_len = len(sys.argv)
    if(arguments_len < 2):
        print('Please pass the argument available option:-s chakra, vishamC')
    elif(arguments_len < 2):
        print('Please pass only required arguments and check README')
    else:
        env_id = sys.argv[2]
        try:
            os.makedirs(env_id)
        except OSError:
            print(env_id,'! already exist')
        roll_obj = policy_gradient()
        roll_obj.train_environment(env_id)
        roll_obj.roll_out()

        batch_size_list = [100, 10]
        alpha_list = [0.1, 0.9]
        gamma_list = [0.1, 0.9]
        iteration = 100
        roll_obj.plot_avg_ret(batch_size_list,alpha_list,gamma_list,iteration)


