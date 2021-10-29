#! /usr/bin/env python3
import meg
import os, argparse
from collections import Counter
import numpy as np
from scipy import stats

## Parse arguments
G = {}
j = 0
for f in np.sort(os.listdir('simulation_main/')):
    A = np.load('simulation_main/'+f,allow_pickle='TRUE').item()
    for index in A:
        G[j] = A[index]
        j += 1

## Repeat initialisations 5 times
nrep = 5
ks_score_em = []; ks_score_ga = []
ks_pval_em = []; ks_pval_ga = []
max_lik_em = np.zeros(j); max_lik_ga = np.zeros(j)
alpha_em = np.zeros((j,2)); beta_em = np.zeros((j,2)); alpha_ga = np.zeros((j,2)); beta_ga = np.zeros((j,2))
mu_em = np.zeros((j,2)); mu_prime_em = np.zeros((j,2)); mu_ga = np.zeros((j,2)); mu_prime_ga = np.zeros((j,2))
phi_em = np.zeros((j,2)); phi_prime_em = np.zeros((j,2)); phi_ga = np.zeros((j,2)); phi_prime_ga = np.zeros((j,2))
for j in G:
    print("\r+++ Iteration {:d} +++".format(j+1), end="")
    ## Set up a MEG model for parameter estimation
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=True, interactions=False, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Seeds for initialisation
    np.random.seed(j)
    seeds = np.random.choice(int(1e6),size=nrep)
    ## Initialise multiple times
    max_lik_em[j] = -1e100
    for s in range(nrep):
        ## Set seed
        np.random.seed(seeds[s])
        ## Initialise the parameter values
        m.alpha = np.random.uniform(low=0, high=1, size=m.n)
        m.beta = np.random.uniform(low=0, high=1, size=m.n)
        m.mu = np.random.uniform(low=0.1, high=1, size=m.n)
        m.mu_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        m.phi = np.random.uniform(low=0.1, high=1, size=m.n)
        m.phi_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        ## Optimise using EM
        l = m.em_optimise(max_iter=100, tolerance=1e-5, niter=1, verbose=False)
        if l[-1] > max_lik_em[j]:
            max_lik_em[j] = l[-1]
            alpha_em[j] = m.alpha
            beta_em[j] = m.beta
            mu_em[j] = m.mu
            mu_prime_em[j] = m.mu_prime
            phi_em[j] = m.phi
            phi_prime_em[j] = m.phi_prime
    ## Calculate p-values
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=True, interactions=False, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Parameter values
    m.alpha = alpha_em[j]; m.beta = beta_em[j]; m.mu = mu_em[j]; m.mu_prime = mu_prime_em[j]; m.phi = phi_em[j]; m.phi_prime = phi_prime_em[j] 
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
        m.alpha = np.random.uniform(low=0, high=1, size=m.n)
        m.beta = np.random.uniform(low=0, high=1, size=m.n)
        m.mu = np.random.uniform(low=0.1, high=1, size=m.n)
        m.mu_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        m.phi = np.random.uniform(low=0.1, high=1, size=m.n)
        m.phi_prime = np.random.uniform(low=0.1, high=1, size=m.n)
        ## Optimise using EM
        for _ in range(10):
            l = m.optimise_meg(prior_penalisation=False, learning_rate=5e-2, method='adam', max_iter=25, verbose=False, tolerance=1e-6, iter_print=False)
        if l[-1] > max_lik_ga[j]:
            max_lik_ga[j] = l[-1]
            alpha_ga[j] = m.alpha
            beta_ga[j] = m.beta
            mu_ga[j] = m.mu
            mu_prime_ga[j] = m.mu_prime
            phi_ga[j] = m.phi
            phi_prime_ga[j] = m.phi_prime
    ## Calculate p-values
    m = meg.meg_model(G[j], tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
    m.specification(main_effects=True, interactions=False, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
    ## Parameter values
    m.alpha = alpha_ga[j]; m.beta = beta_ga[j]; m.mu = mu_ga[j]; m.mu_prime = mu_prime_ga[j]; m.phi = phi_ga[j]; m.phi_prime = phi_prime_ga[j] 
    ## P-value calculations
    m.pvalues()
    pp = [p for x in m.pvals_train.values() for p in list(x)]
    ks_score_ga += [stats.kstest(pp, 'uniform')[0]]
    ks_pval_ga += [stats.kstest(pp, 'uniform')[1]]

## Save output
np.save('simulation_main/loglik_ga.npy', max_lik_ga); np.save('simulation_main/loglik_em.npy', max_lik_em)
np.save('simulation_main/ks_score_ga.npy', ks_score_ga); np.save('simulation_main/ks_score_em.npy', ks_score_em)
np.save('simulation_main/ks_pval_ga.npy', ks_pval_ga); np.save('simulation_main/ks_pval_em.npy', ks_pval_em)
np.save('simulation_main/alpha_ga.npy', alpha_ga); np.save('simulation_main/alpha_em.npy', alpha_em)
np.save('simulation_main/beta_ga.npy', beta_ga); np.save('simulation_main/beta_em.npy', beta_em)
np.save('simulation_main/mu_ga.npy', mu_ga); np.save('simulation_main/mu_em.npy', mu_em)
np.save('simulation_main/mu_prime_ga.npy', mu_prime_ga); np.save('simulation_main/mu_prime_em.npy', mu_prime_em)
np.save('simulation_main/phi_ga.npy', phi_ga); np.save('simulation_main/phi_em.npy', phi_em)
np.save('simulation_main/phi_prime_ga.npy', phi_prime_ga); np.save('simulation_main/phi_prime_em.npy', phi_prime_em)