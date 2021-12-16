#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Random test data
y = np.load('../results/simulation_1/ks_scores.npy')
y2 = np.load('../results/simulation_2/ks_scores.npy')

fig, axes = plt.subplots(figsize=(7,3))

bplot2 = axes.boxplot([y2,y],vert=False,labels=['Simulation 2', 'Simulation 1'], widths=.5, notch=False, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bplot2[element], color='black')

# fill with colors
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

axes.xaxis.grid(True)
axes.set_xlabel('Kolmogorov-Smirnov scores')

plt.savefig('boxplots_simulation.pdf',bbox_inches='tight',pad_inches = 0.1)
plt.show(block=False)