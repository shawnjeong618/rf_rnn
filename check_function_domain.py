import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# hardware parameters (V_DD, etc.)
v_dd = 2.5
v_gnd = 0
v_th = -0.6
w_to_l = 2.0

# simulation parameters
step = 100

# input space
v_in = np.linspace(v_gnd, v_dd, step)

# bias space
v_b = np.linspace(v_th, v_dd, step)

# combine function domain
v_in, v_b = np.meshgrid(v_in, v_b)

# activation condition
condition = v_in > v_b + np.abs(v_th)

#v_out = ((v_in - (v_b + abs(v_th)))**(4/5)) * (((v_b + abs(v_th)) - (15*np.pi*w_to_l*((2*v_in)**(1/2))/16))**(1/5))
#v_out = (((v_b + abs(v_th)) - (15*np.pi*w_to_l*((2*v_in)**(1/2))/16))**(1/5))
#v_out = w_to_l*(v_in - (v_b + abs(v_th)))

# function definition
expr1 = (v_in - (v_b + np.abs(v_th)))
expr1 = np.abs(expr1) ** (4/5)
expr2 = ((15*np.pi*w_to_l*np.sqrt(2*v_in)/16) - (v_b + np.abs(v_th)))
#expr2 = ((v_b + np.abs(v_th))- (15*np.pi*w_to_l*np.sqrt(2*v_in)/16))
expr2 = np.sign(expr2) * np.abs(expr2) ** (1/5)
v_out = expr1 * expr2
v_out[~condition] = 0

# plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(v_in, v_b, v_out, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.set_xlabel(r'$v_{in}$')
ax.set_ylabel(r'$v_b$')
ax.set_zlabel(r'$v_{out}$')
plt.title(r'$v_{out}$ with respect to $v_{b}$ and $v_{in}$')

plt.savefig('3d_plot_orig.png')

# store as csv
