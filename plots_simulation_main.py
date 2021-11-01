#! /usr/bin/env python3
import numpy as np
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Truth
alpha = np.array([0.01,0.05]); beta = np.array([0.07,0.03]); baseline = np.add.outer(alpha,beta)
mu = np.array([0.2,0.15]); mu_prime = np.array([0.1,0.25])
phi = np.array([0.8,0.85]); phi_prime = np.array([0.9,0.75])
decay = mu + phi; decay_prime = mu_prime + phi_prime

## Estimates of baseline
alpha_em = np.load('simulation_main/alpha_em.npy'); beta_em = np.load('simulation_main/beta_em.npy')
alpha_ga = np.load('simulation_main/alpha_ga.npy'); beta_ga = np.load('simulation_main/beta_ga.npy') 
baseline_em = np.array([np.add.outer(alpha_em[i], beta_em[i]) for i in range(alpha_em.shape[0])])
baseline_ga = np.array([np.add.outer(alpha_ga[i], beta_ga[i]) for i in range(alpha_ga.shape[0])])

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
        axs[i,j].set_xlabel('$\\hat{\\alpha}_{'+str(i+1)+'}+\\hat{\\beta}_{'+str(j+1)+'}$')

plt.savefig("simulation_main/alpha_beta.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

## Estimates of jump
mu_em = np.load('simulation_main/mu_em.npy'); mu_prime_em = np.load('simulation_main/mu_prime_em.npy')
mu_ga = np.load('simulation_main/mu_ga.npy'); mu_prime_ga = np.load('simulation_main/mu_prime_ga.npy') 

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        if j == 0:
            mmin = np.min([np.min(mu_em[:,i]),np.min(mu_ga[:,i])])
            mmax = np.max([np.max(mu_em[:,i]),np.max(mu_ga[:,i])])
            positions = np.linspace(mmin,mmax,num=250)
            kernel_em = kde(mu_em[:,i], bw_method='silverman')
            kernel_ga = kde(mu_ga[:,i], bw_method='silverman')
            axs[i,j].hist(mu_em[:,i], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
            axs[i,j].hist(mu_ga[:,i], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
            axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
            axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
            axs[i,j].axvline(x=mu[i],ls='dotted',c='black', lw=3, label='Truth')
            if i == 0 and j == 0:
                axs[i,j].legend()
            axs[i,j].set_ylabel('Density')
            axs[i,j].set_xlabel('$\\hat{\\mu}_{'+str(i+1)+'}$')
        else:
            mmin = np.min([np.min(mu_prime_em[:,i]),np.min(mu_prime_ga[:,i])])
            mmax = np.max([np.max(mu_prime_em[:,i]),np.max(mu_prime_ga[:,i])])
            positions = np.linspace(mmin,mmax,num=250)
            kernel_em = kde(mu_prime_em[:,i], bw_method='silverman')
            kernel_ga = kde(mu_prime_ga[:,i], bw_method='silverman')
            axs[i,j].hist(mu_prime_em[:,i], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
            axs[i,j].hist(mu_prime_ga[:,i], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
            axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
            axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
            axs[i,j].axvline(x=mu_prime[i],ls='dotted',c='black', lw=3, label='Truth')
            if i == 0 and j == 0:
                axs[i,j].legend()
            axs[i,j].set_ylabel('Density')
            axs[i,j].set_xlabel('$\\hat{\\mu}^\\prime_{'+str(i+1)+'}$')

plt.savefig("simulation_main/mu.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

## Estimates of decay
phi_em = np.load('simulation_main/phi_em.npy'); phi_prime_em = np.load('simulation_main/phi_prime_em.npy')
phi_ga = np.load('simulation_main/phi_ga.npy'); phi_prime_ga = np.load('simulation_main/phi_prime_ga.npy') 
decay_em = mu_em + phi_em; decay_prime_em = mu_prime_em + phi_prime_em
decay_ga = mu_ga + phi_ga; decay_prime_ga = mu_prime_ga + phi_prime_ga

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i in [0,1]:
    for j in [0,1]:
        if j == 0:
            mmin = np.min([np.min(decay_em[:,i]),np.min(decay_ga[:,i])])
            mmax = np.max([np.max(decay_em[:,i]),np.max(decay_ga[:,i])])
            positions = np.linspace(mmin,mmax,num=250)
            kernel_em = kde(decay_em[:,i], bw_method='silverman')
            kernel_ga = kde(decay_ga[:,i], bw_method='silverman')
            axs[i,j].hist(decay_em[:,i], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
            axs[i,j].hist(decay_ga[:,i], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
            axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
            axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
            axs[i,j].axvline(x=decay[i],ls='dotted',c='black', lw=3, label='Truth')
            if i == 0 and j == 0:
                axs[i,j].legend()
            axs[i,j].set_ylabel('Density')
            axs[i,j].set_xlabel('$\\hat{\\mu}_{'+str(i+1)+'}+\\hat{\\phi}_{'+str(i+1)+'}$')
        else:
            mmin = np.min([np.min(decay_prime_em[:,i]),np.min(decay_prime_ga[:,i])])
            mmax = np.max([np.max(decay_prime_em[:,i]),np.max(decay_prime_ga[:,i])])
            positions = np.linspace(mmin,mmax,num=250)
            kernel_em = kde(decay_prime_em[:,i], bw_method='silverman')
            kernel_ga = kde(decay_prime_ga[:,i], bw_method='silverman')
            axs[i,j].hist(decay_prime_em[:,i], density=True, bins=15, color='lightgray', histtype=u'step', lw=2)
            axs[i,j].hist(decay_prime_ga[:,i], density=True, bins=15, color='lightgray', histtype=u'step', ls='dashed', lw=2)
            axs[i,j].plot(positions, kernel_em(positions), lw=3, label='EM')
            axs[i,j].plot(positions, kernel_ga(positions), ls='dashed', lw=3, label='Adam')
            axs[i,j].axvline(x=decay_prime[i],ls='dotted',c='black', lw=3, label='Truth')
            if i == 0 and j == 0:
                axs[i,j].legend()
            axs[i,j].set_ylabel('Density')
            axs[i,j].set_xlabel('$\\hat{\\mu}^\\prime_{'+str(i+1)+'}+\\hat{\\phi}^\\prime_{'+str(i+1)+'}$')

plt.savefig("simulation_main/mu_phi.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)

# Import KS scores
ks_em = np.load('simulation_main/ks_score_em.npy')
ks_ga = np.load('simulation_main/ks_score_ga.npy')
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

plt.savefig("simulation_main/ks_scores.png", bbox_inches='tight', pad_inches = 0.1, dpi=500)
plt.show(block=False)