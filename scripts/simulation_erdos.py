#! /usr/bin/env python3
import meg
import os, argparse
from collections import Counter
import numpy as np

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
graphs = {}

## Simulate graphs
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
    if m.main_effects:
        m.alpha = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        m.beta = np.random.uniform(low=1e-5, high=1e-4, size=m.n)
        if not poisson_main_effects:
            m.mu = np.random.uniform(low=1e-2, high=1e-1, size=m.n)
            m.mu_prime = np.random.uniform(low=1e-2, high=1e-1, size=m.n) 
            m.phi = np.random.uniform(low=1e-2, high=1e-1, size=m.n)
            m.phi_prime = np.random.uniform(low=1e-2, high=1e-1, size=m.n)
    if m.interactions:
        if m.D == 1:
            m.gamma = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            m.gamma_prime = np.random.uniform(low=1e-5, high=1e-1, size=m.n)
            if not poisson_interactions:
                m.nu = np.random.uniform(low=1e-2, high=1e0, size=m.n)
                m.nu_prime = np.random.uniform(low=1e-2, high=1e0, size=m.n)
                m.theta = 1 - m.nu # np.random.uniform(low=1e-2, high=1e0, size=m.n)
                m.theta_prime = 1 - m.nu_prime # np.random.uniform(low=1e-2, high=1e0, size=m.n)
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
    graphs[q] = G

np.save(dest_folder + '/graphs.npy', graphs)