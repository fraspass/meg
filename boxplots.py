#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Random test data
y = np.loadtxt('simulation_1/ks_scores.csv')
y2 = np.loadtxt('simulation_2/ks_scores.csv')

fig, axes = plt.subplots(figsize=(7,3))

bplot2 = axes.boxplot([y2,y],vert=False,labels=['Simulation 2', 'Simulation 1'], widths=.5, notch=True, patch_artist=True)

# fill with colors
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

axes.xaxis.grid(True)
axes.set_xlabel('Kolmogorov-Smirnov scores')

plt.savefig("boxplots_simulation.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()