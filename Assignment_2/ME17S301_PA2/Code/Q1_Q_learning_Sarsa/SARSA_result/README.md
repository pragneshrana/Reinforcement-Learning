# Control Algorithms:
Implementation Puddle Gridworld and Policy Gradient algorith in python

# Implemention Refrence :
Implemetation is done using refrence of openAI gym and Sutton and Barto book.
 
* Q-learning
* SARSA
* SRASA-Lambda
* Policy Gradients
* Epsilon-Greedy Algorithm

# Description of the Puddle World :
Pudle world is typical grid world in which each action is stochastic action. The desired action acrtion has probability 0.9 apart from that other action has 0.1. It contains GridWorld-A,B,C.  All three world has different terminal states, in which world-C has westerly blowing.Terminal State has reward +10. Middle part contains puddle, as depth of puddle increases it gives negative reward.

To simulate this problem Q-learning,SARSA,SARSA-LAMBDA method is used with epsilon-greedy policy.

# Simulate PuddleWorld:
Run command below,

python SimulateGridWorld.py -s 'method_type'

Available method type is,
method_type = {QL, SARSA, SARSA_LAMBDA}



# Description of the Policy Gradient :
For Policy Gradient , Initial task is to implement the chakra and vishamC world. The main differnece between two world is reward function. 

# Simulate using Policy Gradient:
Run command below,

python policy_gradient.py -s 'environmentID'

Available method type is,
environmentID = {chakra, vishamC}
# File Directory:

# Dependencie

* numpy
* matplotlib
* OpenAI GYM


