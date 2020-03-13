# For Plotting State Value from trained policy 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def ValueFuncPlot(method_type):
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
    state_space = np.load('StateSpace'+str(method_type)+'.npy')
    val = np.load('ValueFunc'+str(method_type)+'.npy')
    val = np.around(val, decimals=2)
    x = []
    y = []
    for j in range(len(state_space)):
        x.append(state_space[j][0])
        y.append(state_space[j][1])
    plt.scatter(x, y,marker='o')

    for i, txt in enumerate(val):
        plt.annotate(txt, (x[i], y[i]), ha='center', size='large')
    plt.title('Value Function for '+str(method_type)+'after learning')
    plt.xlabel('X val ')
    plt.ylabel('Y val')
    plt.savefig('Val_Func_'+str(method_type)+'.png', dpi=300)
    plt.show()

