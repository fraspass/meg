#! /usr/bin/env python3
import meg
import os, argparse
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
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
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="simulation", const=True, nargs="?",\
    help="String: name of the destination folder for the output files.")
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
## Dimension of the latent features
parser.add_argument("-d", type=int, dest="d", default=1, const=True, nargs="?",\
	help="Integer: number of latent features. Default: d=1.")
parser.add_argument("-n", type=int, dest="n", default=20, const=True, nargs="?",\
	help="Integer: number of nodes. Default: n=20.")
parser.add_argument("-T", type=int, dest="T", default=1000000, const=True, nargs="?",\
	help="Integer: maximum time of simulation. Default: T=1000000.")
parser.add_argument("-M", type=int, dest="M", default=10000, const=True, nargs="?",\
	help="Integer: number of simulated events. Default: M=10000.")
parser.add_argument("-p", type=float, dest="p", default=.25, const=True, nargs="?",\
	help="Integer: probability of connection. Default: p=0.5")

## Parse arguments
args = parser.parse_args()
dest_folder = args.dest_folder
main_effects = True if args.main_effects else False
interactions = True if args.interactions else False
poisson_main_effects = True if args.poisson_main_effects else False
poisson_interactions = True if args.poisson_interactions else False
hawkes_main_effects = True if args.hawkes_main_effects else False
hawkes_interactions = True if args.hawkes_interactions else False
D = args.d
n = args.n
T_sim = args.T
p = args.p
M = args.M

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Set the seed
np.random.seed(117)

## Build the MEG model
ks_scores = {}
ks_pvals = {}

for q in range(100):
    ## Simulate adjacency matrix
    A = {}
    for i in range(n-1):
        for j in range(i,n):
            if np.random.uniform(size=1) < p:
                A[i,j] = [1]
            if np.random.uniform(size=1) < p:
                A[j,i] = [1]
    m = meg.meg_model(A, tau_zero=True, verbose=False, discrete=False, force_square=True)
    m.specification(main_effects=main_effects, interactions=interactions, 
                poisson_me=poisson_main_effects, poisson_int=poisson_interactions,
                hawkes_me=hawkes_main_effects, hawkes_int=hawkes_interactions, 
                D=D, verbose=False)
    m.prior_initialisation()
    ## Initialise all to the same values
    m.alpha = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
    m.beta = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
    if not poisson_main_effects:
        m.mu = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
        m.mu_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n) 
        m.phi = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
        m.phi_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
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
    ## Simulate data
    m.simulate(T=T_sim, m=M, copy_dict=False, verbose=True)
    G = m.A
    ks_scores[q] = []
    ks_pvals[q] = []
    for _ in range(1):
        print('Simulation ', q, '\t', 'Initialisation ', _+1, sep='')
        m2 = meg.meg_model(G, T=T_sim, tau_zero=True, verbose=False, discrete=False, force_square=True)
        m2.specification(main_effects=main_effects, interactions=interactions, 
                    poisson_me=poisson_main_effects, poisson_int=poisson_interactions,
                    hawkes_me=hawkes_main_effects, hawkes_int=hawkes_interactions, 
                    D=D, verbose=False)
        m2.prior_initialisation()
        ## Initialise all to the same values
        m2.alpha = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        m2.beta = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        if not poisson_main_effects:
            m2.mu = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
            m2.mu_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n) 
            m2.phi = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
            m2.phi_prime = np.random.uniform(low=1e-5, high=1e-3, size=m.n)
        if m2.D == 1:
            m2.gamma = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            m2.gamma_prime = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            if not poisson_interactions:
                m2.nu = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m2.nu_prime = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m2.theta = np.random.uniform(low=1e-2, high=1e1, size=m.n)
                m2.theta_prime = np.random.uniform(low=1e-2, high=1e1, size=m.n)
        else:
            m2.gamma = np.random.uniform(low=1e-5, high=1e-1, size=(m.n,m.D))
            m2.gamma_prime = np.random.uniform(low=1e-5, high=1e-1, size=(m.n,m.D))
            if not poisson_interactions:
                m2.nu = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m2.nu_prime = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m2.theta = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
                m2.theta_prime = np.random.uniform(low=1e-5, high=1e-0, size=(m.n,m.D))
        ## Fit model to the data 
        l = m2.optimise_meg(prior_penalisation=False, learning_rate=0.1, method='adam', max_iter=250, verbose=False, tolerance=1e-5)
        ## Calculate the p-values
        m2.pvalues()
        pp = [p for x in m2.pvals_train.values() for p in list(x)]
        ## Calculate the KS scores
        ks_scores[q] += [stats.kstest(pp, 'uniform')[0]]
        ks_pvals[q] += [stats.kstest(pp, 'uniform')[1]]

## Save files
scores = [p for x in ks_scores.values() for p in list(x)]
pvals = [p for x in ks_pvals.values() for p in list(x)]

## Save files
np.savetxt(dest_folder + '/ks_scores.csv', scores, fmt='%.20f', delimiter=',')
np.savetxt(dest_folder + '/ks_pvals.csv', pvals, fmt='%.20f', delimiter=',')