#! /usr/bin/env python3
import numpy as np
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Truth
gamma = np.array([0.1,0.5]); gamma_prime = np.array([0.1,0.3]); baseline = np.outer(gamma, gamma_prime)
nu = np.array([0.8,0.4]); nu_prime = np.array([0.5,0.25]); jump = np.outer(nu, nu_prime)
theta = np.array([0.2,0.6]); theta_prime = np.array([0.5,0.75]); decay = np.outer(nu + theta, nu_prime + theta_prime)

## Estimates of baseline
gamma_em = np.load('simulation_inter/gamma_em.npy'); gamma_prime_em = np.load('simulation_inter/gamma_prime_em.npy')
gamma_ga = np.load('simulation_inter/gamma_ga.npy'); gamma_prime_ga = np.load('simulation_inter/gamma_prime_ga.npy') 
baseline_em = np.array([np.outer(gamma_em[i], gamma_prime_em[i]) for i in range(gamma_em.shape[0])])
baseline_ga = np.array([np.outer(gamma_ga[i], gamma_prime_ga[i]) for i in range(gamma_ga.shape[0])])

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        mmin = np.min([np.min(baseline_em[:,i,j]),np.min(baseline_ga[:,i,j])])
        mmax = np.max([np.max(baseline_em[:,i,j]),np.max(baseline_ga[:,i,j])])
        positions = np.linspace(mmin,mmax,num=250)
        kernel_em = kde(baseline_em[:,i,j], bw_method='silverman')
        kernel_ga = kde(baseline_ga[:,i,j], bw_method='silverman')
        axs[i,j].hist(baseline_em[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
        axs[i,j].hist(baseline_ga[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
        axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
        axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
        axs[i,j].axvline(x=baseline[i,j],ls='dotted',c='black', lw=3, label='Truth')
        if i == 0 and j == 0:
            axs[i,j].legend()
        axs[i,j].set_ylabel('Density')
        axs[i,j].set_xlabel('$\\hat{\\gamma}_{'+str(i+1)+'}\\hat{\\gamma}^\\prime_{'+str(j+1)+'}$')

plt.savefig("simulation_inter/gamma.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

## Estimates of jump
nu_em = np.load('simulation_inter/nu_em.npy'); nu_prime_em = np.load('simulation_inter/nu_prime_em.npy')
nu_ga = np.load('simulation_inter/nu_ga.npy'); nu_prime_ga = np.load('simulation_inter/nu_prime_ga.npy') 
jump_em = np.array([np.outer(nu_em[i], nu_prime_em[i]) for i in range(nu_em.shape[0])])
jump_ga = np.array([np.outer(nu_ga[i], nu_prime_ga[i]) for i in range(nu_ga.shape[0])])

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        mmin = np.min([np.min(jump_em[:,i,j]),np.min(jump_ga[:,i,j])])
        mmax = np.max([np.max(jump_em[:,i,j]),np.max(jump_ga[:,i,j])])
        positions = np.linspace(mmin,mmax,num=250)
        kernel_em = kde(jump_em[:,i,j], bw_method='silverman')
        kernel_ga = kde(jump_ga[:,i,j], bw_method='silverman')
        axs[i,j].hist(jump_em[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
        axs[i,j].hist(jump_ga[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
        axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
        axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
        axs[i,j].axvline(x=jump[i,j],ls='dotted',c='black', lw=3, label='Truth')
        if i == 0 and j == 0:
            axs[i,j].legend()
        axs[i,j].set_ylabel('Density')
        axs[i,j].set_xlabel('$\\hat{\\nu}_{'+str(i+1)+'}\\hat{\\nu}^\\prime_{'+str(j+1)+'}$')

plt.savefig("simulation_inter/nu.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

## Estimates of decay
theta_em = np.load('simulation_inter/theta_em.npy'); theta_prime_em = np.load('simulation_inter/theta_prime_em.npy')
theta_ga = np.load('simulation_inter/theta_ga.npy'); theta_prime_ga = np.load('simulation_inter/theta_prime_ga.npy') 
decay_em = np.array([np.outer(nu_em[i] + theta_em[i], nu_prime_em[i] + theta_prime_em[i]) for i in range(nu_em.shape[0])])
decay_ga = np.array([np.outer(nu_ga[i] + theta_ga[i], nu_prime_ga[i] + theta_prime_ga[i]) for i in range(nu_ga.shape[0])])

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        mmin = np.min([np.min(decay_em[:,i,j]),np.min(decay_ga[:,i,j])])
        mmax = np.max([np.max(decay_em[:,i,j]),np.max(decay_ga[:,i,j])])
        positions = np.linspace(mmin,mmax,num=250)
        kernel_em = kde(decay_em[:,i,j], bw_method='silverman')
        kernel_ga = kde(decay_ga[:,i,j], bw_method='silverman')
        axs[i,j].hist(decay_em[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
        axs[i,j].hist(decay_ga[:,i,j], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
        axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
        axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
        axs[i,j].axvline(x=decay[i,j],ls='dotted',c='black', lw=3, label='Truth')
        if i == 0 and j == 0:
            axs[i,j].legend()
        axs[i,j].set_ylabel('Density')
        axs[i,j].set_xlabel('$(\\hat{\\nu}_{'+str(i+1)+'}+\\hat{\\theta}_{'+str(i+1)+'})(\\hat{\\nu}^\\prime_{'+str(j+1)+'}+\\hat{\\theta}^\\prime_{'+str(j+1)+'})$')

plt.savefig("simulation_inter/nu_theta.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

# Import KS scores
ks_em = np.load('simulation_inter/ks_score_em.npy')
ks_ga = np.load('simulation_inter/ks_score_ga.npy')
fig, axes = plt.subplots()
bplot = axes.boxplot([ks_ga,ks_em],vert=False,labels=['Adam', 'EM'], widths=.5, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bplot[element], color='black')

axes.xaxis.grid(True)
axes.set_xlabel('Kolmogorov-Smirnov scores')
# Fill with colors
colors = ['orange', 'cornflowerblue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig("simulation_inter/ks_scores.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)