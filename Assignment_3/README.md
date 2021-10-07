# HRL and DQN:
Implementation Hierarchical and Deep Reinforcement Learning in python

# Implemention Refrence :
Implemetation is done using refrence of Sutton and Barto book and online resources as given in report.
 
* Hierarchical Reinforcement Learning
* Deep Reinforcement Learning



# Description of the HRL :
The four-room grid world is given. The goal is to train the agent such that following optimal policy agent can reach the goal G1 and G2. SMDP and Intra-option Q learning are used to solve this problem.

# Run HRL:
Run command below,

python Q1.py 'method_type'

Available method type is,
method_type = {learn_policy, smdp, intra, plots}

* First obtain the policy by running method 'learn_policy'
* then use your favorite method 'smdp' or 'intra' to combine with options
* To generate plots use method 'plots'




# Description of the DQN :
The deep Q learning has been implemented on standard cart-pole problem. The goal is to stabilize the Reinforcement Learning and make observation about different parameters.

# Run DQN:
Run command below,

python 'File_name'.py

File_name = {DQN_cartpole, epsilon_DQN, Hidden_layer_DQN, minibatch_DQN, target_frequency_DQN, without_reply, without_target, without_target_reply_DQN, combined_plot_for_without}

* First generate result in case of 'without_reply' , 'without_target', 'without_target_reply_DQN' and then generate plots using 'combined_plot_for_without'


# File Directory:
-ME17S301_PA3

-- Solution
--- Q1_SMDP_Intra
---- code
------ gym_gridworld
------ Q_saves
------ Q1.py
---- result - all the result files as given in report


--- Q2_Policy_Gradient
------ CODE_N_RESULT - this folder contains all the code and result as given in report
----------- DQN_cartpole.py
----------- epsilon_DQN.py
----------- Hidden_layer_DQN.py
----------- minibatch_DQN.py
----------- target_frequency_DQN.py
----------- without_reply.py
----------- without_target.py
----------- without_target_reply_DQN.py
----------- combined_plot_for_without.py
----------- OTHER RESULT FILES/FOLDERS


-- Report_ME17S301_PA3.pdf

-- README
# Dependencie

* numpy
* matplotlib
* OpenAI GYM
* TensorFlow
* scikit learn

