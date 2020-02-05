# 10-Arm testbed
Implementation multi-armed bandits algorithms using Python 3.0 +.

# Implemention Refrence 
Algorithms are implemented on a 10-arm testbed, as described in by Richard and Sutton.

* Epsilon-Greedy Algorithm
* Softmax Algorithm
* Upper Confidence Bound(UCB1)


# Description of the Experiment

The testbed contains 2000 bandit problems with 10 arms each, with the true action values for each arm in each bandit sampled from a normal distribution N(0,1). When a learning algorithm is applied to any of these arms at time t, action is selected from each bandit problem and it gets a reward which is sampled from respective arm distribution. The performance of each of the implemented algorithms is measured as ”Average Reward” (or) ”% Optimal Actions picked” at each time step.


# Implementations

The code is in ./code/experimental_testbed_files
Each .py files is answer if each assignment question.
For question-5 there are 3 files.

The 10-arm testbed is implemented separately for each algorithm in the corresponding code file. The epsilon-greedy, softmax and UCB1 algorithms have been implemented for 1000 (time) steps each with varying values of respective parameters. The procedure was repeated for 2000 independent runs, each time with a different bandit problem. The rewards obtained over all these runs is averaged to get an idea of the algorithm's average behaviour.

Comparison graphs have been plotted and can be found plotting folders.

More descriptions can be found in the respective folders.

# Dependencie

* numpy
* matplotlib

