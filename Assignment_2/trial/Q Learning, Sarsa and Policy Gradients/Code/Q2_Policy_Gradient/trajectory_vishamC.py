# For plotting Policy trajectories of learned agent

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

#generating mesh grids
x_values = np.linspace(-1.0, 1.0, 100)
y_values = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = (.5*X**2 + 5*Y**2)

#ploting figures
plt.figure()
cp = plt.contour(X, Y, Z)
plt.clabel(cp, inline=True,fontsize=10)
for i in range(15):
    tmp = np.load("vishamC_traj_"+str(i)+".npy")
    x = []
    y = []
    for j in range(len(tmp)):
        x.append(tmp[j][0])
        y.append(tmp[j][1])
    plt.plot(x, y)
plt.title('Policy trajectories of learned agent for vishamC')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("vishamC_traj.png", dpi=300)
# plt.show()
