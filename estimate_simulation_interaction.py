#! /usr/bin/env python3
import meg
import os, argparse
from collections import Counter
import numpy as np
from scipy import stats

## Parse arguments
G = {}
j = 0
for f in np.sort(os.listdir('simulation_inter/')):
    A = np.load('simulation_inter/'+f,allow_pickle='TRUE').item()
    for index in A:
        G[j] = A[index]
        j += 1

## Repeat initialisations 5 times
nrep = 5
ks_score_em = []; ks_score_ga = []
ks_pval_em = []; ks_pval_ga = []
max_lik_em = np.zeros(j); max_lik_ga = np.zeros(j)
gamma_em = np.zeros((j,2)); gamma_prime_em = np.zeros((j,2)); gamma_ga = np.zeros((j,2)); gamma_prime_ga = np.zeros((j,2))
nu_em = np.zeros((j,2)); nu_prime_em = np.zeros((j,2)); nu_ga = np.zeros((j,2)); nu_prime_ga = np.zeros((j,2))
theta_em = np.zeros((j,2)); theta_prime_em = np.zeros((j,2)); theta_ga = np.zeros((j,2)); theta_prime_ga = np.zeros((j,2))
for j in G:
    print("\r+++ Iteration {:d} +++".format(j+1), end="")
    ## Set up a MEG model for parameter estimation
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=False, interactions=True, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Seeds for initialisation
    np.random.seed(j)
    seeds = np.random.choice(int(1e6),size=nrep)
    ## Initialise multiple times
    max_lik_em[j] = -1e100
    for s in range(nrep):
        ## Set seed
        np.random.seed(seeds[s])
        ## Initialise the parameter values
        m.gamma = np.random.uniform(low=0, high=1, size=m.n)
        m.gamma_prime = np.random.uniform(low=0, high=1, size=m.n)
        m.nu = np.random.uniform(low=0.1, high=1, size=m.n)
        m.nu_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        m.theta = np.random.uniform(low=0.1, high=1, size=m.n)
        m.theta_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        ## Optimise using EM
        l = m.em_optimise(max_iter=100, tolerance=1e-5, niter=1, verbose=False)
        if l[-1] > max_lik_em[j]:
            max_lik_em[j] = l[-1]
            gamma_em[j] = m.gamma
            gamma_prime_em[j] = m.gamma_prime
            nu_em[j] = m.nu
            nu_prime_em[j] = m.nu_prime
            theta_em[j] = m.theta
            theta_prime_em[j] = m.theta_prime
    ## Calculate p-values
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=False, interactions=True, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Parameter values
    m.gamma = gamma_em[j]; m.gamma_prime = gamma_prime_em[j]; m.nu = nu_em[j]; m.nu_prime = nu_prime_em[j]; m.theta = theta_em[j]; m.theta_prime = theta_prime_em[j] 
    ## P-value calculations
    m.pvalues()
    pp = [p for x in m.pvals_train.values() for p in list(x)]
    ks_score_em += [stats.kstest(pp, 'uniform')[0]]
    ks_pval_em += [stats.kstest(pp, 'uniform')[1]]
    ## Repeat for Adam (gradient ascent)
    max_lik_ga[j] = -1e100
    for s in range(nrep):
        ## Set seed for *same* initialisation
        np.random.seed(seeds[s])
        ## Initialise the parameter values
        m.gamma = np.random.uniform(low=0, high=1, size=m.n)
        m.gamma_prime = np.random.uniform(low=0, high=1, size=m.n)
        m.nu = np.random.uniform(low=0.1, high=1, size=m.n)
        m.nu_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        m.theta = np.random.uniform(low=0.1, high=1, size=m.n)
        m.theta_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        ## Optimise using EM
        for _ in range(10):
            l = m.optimise_meg(prior_penalisation=False, learning_rate=5e-2, method='adam', max_iter=25, verbose=False, tolerance=1e-6, iter_print=False)
        if l[-1] > max_lik_ga[j]:
            max_lik_ga[j] = l[-1]
            gamma_ga[j] = m.gamma
            gamma_prime_ga[j] = m.gamma_prime
            nu_ga[j] = m.nu
            nu_prime_ga[j] = m.nu_prime
            theta_ga[j] = m.theta
            theta_prime_ga[j] = m.theta_prime
    ## Calculate p-values
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=False, interactions=True, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Parameter values
    m.gamma = gamma_ga[j]; m.gamma_prime = gamma_prime_ga[j]; m.nu = nu_ga[j]; m.nu_prime = nu_prime_ga[j]; m.theta = theta_ga[j]; m.theta_prime = theta_prime_ga[j] 
    ## P-value calculations
    m.pvalues()
    pp = [p for x in m.pvals_train.values() for p in list(x)]
    ks_score_ga += [stats.kstest(pp, 'uniform')[0]]
    ks_pval_ga += [stats.kstest(pp, 'uniform')[1]]

## Save output
np.save('simulation_main/loglik_ga.npy', max_lik_ga); np.save('simulation_main/loglik_em.npy', max_lik_em)
np.save('simulation_main/ks_score_ga.npy', ks_score_ga); np.save('simulation_main/ks_score_em.npy', ks_score_em)
np.save('simulation_main/ks_pval_ga.npy', ks_pval_ga); np.save('simulation_main/ks_pval_em.npy', ks_pval_em)
np.save('simulation_main/gamma_ga.npy', gamma_ga); np.save('simulation_main/gamma_em.npy', gamma_em)
np.save('simulation_main/gamma_prime_ga.npy', gamma_prime_ga); np.save('simulation_main/gamma_prime_em.npy', gamma_prime_em)
np.save('simulation_main/nu_ga.npy', nu_ga); np.save('simulation_main/nu_em.npy', nu_em)
np.save('simulation_main/nu_prime_ga.npy', nu_prime_ga); np.save('simulation_main/nu_prime_em.npy', nu_prime_em)
np.save('simulation_main/theta_ga.npy', theta_ga); np.save('simulation_main/theta_em.npy', theta_em)
np.save('simulation_main/theta_prime_ga.npy', theta_prime_ga); np.save('simulation_main/theta_prime_em.npy', theta_prime_em)