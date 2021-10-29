#! /usr/bin/env python3
import meg
import os, argparse
import numpy as np
from scipy import stats

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="simulation_inter", const=True, nargs="?",\
    help="String: name of the destination folder for the output files.")
parser.add_argument("-T", type=int, dest="T", default=10000000, const=True, nargs="?",\
	help="Integer: maximum time of simulation. Default: T=10000000.")
parser.add_argument("-M", type=int, dest="M", default=3000, const=True, nargs="?",\
	help="Integer: number of simulated events. Default: M=3000.")
parser.add_argument("-N", type=int, dest="N", default=10, const=True, nargs="?",\
	help="Integer: number of simulated graphs. Default: N=10.")
parser.add_argument("-S", type=int, dest="S", default=117, const=True, nargs="?",\
	help="Integer: seed for the simulation. Default: S=117.")

## Parse arguments
args = parser.parse_args()
dest_folder = args.dest_folder
n = 2
T_sim = args.T
M = args.M
nsim = args.N
S = args.S

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Set the seed
np.random.seed(117)

## Build the MEG model
ks_scores = {}
ks_pvals = {}

## Empty matrix
A = {}
for i in range(n):
    for j in range(n):
        A[i,j] = [1]

## MEG model setup for simulation
m = meg.meg_model(A, tau_zero=True, full_links=True, verbose=False, discrete=False, force_square=True, evaluate_directed=False)
m.specification(main_effects=False, interactions=True, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False)
m.prior_initialisation()

## True values of the parameters
m.gamma = np.array([0.1,0.5])
m.gamma_prime = np.array([0.1,0.3])
m.nu = np.array([0.8,0.4])
m.nu_prime = np.array([0.5,0.25])
m.theta = np.array([0.2,0.6])
m.theta_prime = np.array([0.5,0.75])

## Simulate 
np.random.seed(S)
G = {}
for i in range(nsim):
    m.simulate(T=T_sim, m=M, copy_dict=False, verbose=True)
    G[i] = m.A

## Save dictionary
np.save(dest_folder + '/meg_simulate_' + S + '.npy', G) 