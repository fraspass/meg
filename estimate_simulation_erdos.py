#! /usr/bin/env python3
import meg
import os, glob, argparse
from collections import Counter
import numpy as np
from scipy import stats

## Initialise the parameters
main_effects = False
interactions = False
poisson_main_effects = False
poisson_interactions = False
hawkes_main_effects = False
hawkes_interactions = False

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="folder", default="simulation_1", const=True, nargs="?",\
    help="String: name of the folder for the input files.")
parser.add_argument("-m", action='store_true', dest="main_effects", default=main_effects,
	help="Boolean variable for the main effects, default FALSE.")
parser.add_argument("-i", action='store_true', dest="interactions", default=interactions,
	help="Boolean variable for the interactions, default FALSE.")
parser.add_argument("-pm", action='store_true', dest="poisson_main_effects", default=poisson_main_effects,
	help="Boolean variable for the Poisson process for the main effects, default FALSE.")
parser.add_argument("-pi", action='store_true', dest="poisson_interactions", default=poisson_interactions,
	help="Boolean variable for the Poisson process for the interactions, default FALSE.")
parser.add_argument("-hm", action='store_true', dest="hawkes_main_effects", default=hawkes_main_effects,
	help="Boolean variable for the Hawkes process for the main effects, default FALSE. Alternatively, the Wald-Markov process is used.")
parser.add_argument("-hi", action='store_true', dest="hawkes_interactions", default=hawkes_interactions,
	help="Boolean variable for the Hawkes process for the interactions, default FALSE. Alternatively, the Wald-Markov process is used.")
parser.add_argument("-d", type=int, dest="d", default=1, const=True, nargs="?",\
	help="Integer: number of latent features. Default: d=1.")
parser.add_argument("-eta", type=float, dest="eta", default=1e-3, const=True, nargs="?",\
	help="Float: number of latent features. Default: 0.001.")

## Parse arguments
args = parser.parse_args()
input_folder = args.folder
main_effects = True if args.main_effects else False
interactions = True if args.interactions else False
poisson_main_effects = True if args.poisson_main_effects else False
poisson_interactions = True if args.poisson_interactions else False
hawkes_main_effects = True if args.hawkes_main_effects else False
hawkes_interactions = True if args.hawkes_interactions else False
D = args.d
eta = args.eta

## Parse arguments
G = {}
j = 0
A = np.load(input_folder + '/graphs.npy', allow_pickle='TRUE').item()
for index in A:
    G[j] = A[index]
    j += 1

## Model
m = meg.meg_model(G[0], tau_zero=True, verbose=False, discrete=False, force_square=True)
## Extract n
N_nodes = m.n

## Repeat initialisations 5 times
ks_scores = []; ks_pvals = []
if main_effects:
    alpha = np.zeros((j,N_nodes)); beta = np.zeros((j,N_nodes))
    if not poisson_main_effects:
        mu = np.zeros((j,N_nodes)); phi = np.zeros((j,N_nodes))
        mu_prime = np.zeros((j,N_nodes)); phi_prime = np.zeros((j,N_nodes))

if interactions:
    if D == 1:
        gamma = np.zeros((j,N_nodes)); gamma_prime = np.zeros((j,N_nodes))
        nu = np.zeros((j,N_nodes)); nu_prime = np.zeros((j,N_nodes))
        theta = np.zeros((j,N_nodes)); theta_prime = np.zeros((j,N_nodes))
    else:
        gamma = np.zeros((j,N_nodes,D)); gamma_prime = np.zeros((j,N_nodes,D))
        nu = np.zeros((j,N_nodes,D)); nu_prime = np.zeros((j,N_nodes,D))
        theta = np.zeros((j,N_nodes,D)); theta_prime = np.zeros((j,N_nodes,D))
        
for j in G:
    print("\r+++ Graph {:d} +++".format(j+1), end="")
    ## Set up a MEG model for parameter estimation
    m = meg.meg_model(G[j], tau_zero=True, verbose=False, discrete=False, force_square=True)
    m.specification(main_effects=main_effects, interactions=interactions, 
                        poisson_me=poisson_main_effects, poisson_int=poisson_interactions,
                        hawkes_me=hawkes_main_effects, hawkes_int=hawkes_interactions, 
                        D=D, verbose=False)
    np.random.seed(j)
    ## Initialise the parameter values
    m.prior_initialisation()
    ## Initialise all to the same values
    if m.main_effects:
        m.alpha = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        m.beta = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        if not poisson_main_effects:
            m.mu = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
            m.mu_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n) 
            m.phi = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
            m.phi_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
    if m.interactions:
        if m.D == 1:
            m.gamma = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            m.gamma_prime = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            if not poisson_interactions:
                m.nu = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m.nu_prime = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m.theta = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m.theta_prime = np.random.uniform(low=1e-2, high=1e1, size=m.n)
        else:
            m.gamma = np.random.uniform(low=1e-5, high=1e-1, size=(m.n,m.D))
            m.gamma_prime = np.random.uniform(low=1e-5, high=1e-1, size=(m.n,m.D))
            if not poisson_interactions:
                m.nu = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m.nu_prime = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m.theta = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m.theta_prime = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
    ## Optimise using EM
    l = m.optimise_meg(prior_penalisation=False, learning_rate=eta, method='adam', max_iter=500, verbose=False, tolerance=1e-6, iter_print=False)
    ## Store output
    if m.main_effects:
        alpha[j] = np.copy(m.alpha); beta[j] = np.copy(m.beta)
        if not m.poisson_me:
            mu[j] = np.copy(m.mu); phi[j] = np.copy(m.phi)
            mu_prime[j] = np.copy(m.mu_prime); phi_prime[j] = np.copy(m.phi_prime)
    if m.interactions:
        gamma[j] = np.copy(m.gamma); gamma_prime[j] = np.copy(m.gamma_prime)
        if not m.poisson_int:
            nu[j] = np.copy(m.nu); theta = np.copy(m.theta)
            nu_prime[j] = np.copy(m.nu_prime); theta_prime = np.copy(m.theta_prime)
    ## Calculate the p-values
    m.pvalues()
    pp = [p for x in m.pvals_train.values() for p in list(x)]
    ## Calculate the KS scores
    ks_scores += [stats.kstest(pp, 'uniform')[0]]
    ks_pvals += [stats.kstest(pp, 'uniform')[1]]

## Save files
np.save(input_folder + '/ks_scores.npy', ks_scores)
np.save(input_folder + '/ks_pvals.npy', ks_pvals)
if main_effects:
    np.save(input_folder + '/alpha.npy', alpha); np.save(input_folder + '/beta.npy', beta)
    if not poisson_main_effects:
        np.save(input_folder + '/mu.npy', mu); np.save(input_folder + '/phi.npy', phi)
        np.save(input_folder + '/mu_prime.npy', mu_prime); np.save(input_folder + '/phi_prime.npy', phi_prime)

if interactions:
    np.save(input_folder + '/gamma.npy', gamma); np.save(input_folder + '/gamma_prime.npy', gamma_prime)
    if not poisson_interactions:
        np.save(input_folder + '/nu.npy', nu); np.save(input_folder + '/theta.npy', theta) 
        np.save(input_folder + '/nu_prime.npy', nu_prime); np.save(input_folder + '/theta_prime.npy', theta_prime)