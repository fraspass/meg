#! /usr/bin/env python3
import meg
import os, argparse
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix

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
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="icl_results", const=True, nargs="?",\
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

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Bipartite graph
test = True
G = {}
if test:
    G_test = {}

huxley_map = {}
server_map = {}
huxley_inverse_map = {}
server_inverse_map = {}

k_comp = 0
k_host = 0
tmin = 1579478400 ## Monday January 20, 2020 (Unix time)
with open('data/huxley_filter.csv') as f:
    for line in f:
        line = line.rstrip('\r\n').split(',')
        link = tuple(line[:2])
        comp = link[0]
        host = link[1]
        ## Remove repeated observations
        times = np.sort(np.unique([float(x) for x in line[2:]]))
        times = times[times > tmin] - tmin
        times_train = times[times < 86400 * 14]    
        if len(times_train) > 1: 
            if comp not in huxley_map:
                huxley_map[comp] = k_comp
                huxley_inverse_map[k_comp] = comp
                k_comp += 1
            if host not in server_map:
                server_map[host] = k_host
                server_inverse_map[k_host] = host
                k_host += 1
            ## Add link to the graph
            if (huxley_map[comp], server_map[host]) in G:
                if test and len(times_train) > 0:
                    G[huxley_map[comp], server_map[host]] += list(times_train)
                if not test:
                    G[huxley_map[comp], server_map[host]] += list(times)
            else:
                if test and len(times_train) > 0:
                    G[huxley_map[comp], server_map[host]] = list(times_train) 
                if not test: 
                    G[huxley_map[comp], server_map[host]] = list(times)

if test:
    with open('data/huxley_filter.csv') as f:
        for line in f:
            line = line.rstrip('\r\n').split(',')
            link = tuple(line[:2])
            comp = link[0]
            host = link[1]
            ## Remove repeated observations
            times = np.sort(np.unique([float(x) for x in line[2:]]))
            times = times[times > tmin] - tmin
            times_test = times[np.logical_and(times >= 86400 * 14, times < 86400 * 21)]
            if comp in huxley_map and host in server_map and len(times_test) > 0:
                if (huxley_map[comp], server_map[host]) in G_test:
                    G_test[huxley_map[comp], server_map[host]] += list(times_test)
                else:
                    G_test[huxley_map[comp], server_map[host]] = list(times_test)

## Build the MEG model
m = meg.meg_model(G, verbose=False) 
m.specification(main_effects=main_effects, interactions=interactions, 
                poisson_me=poisson_main_effects, poisson_int=poisson_interactions,
                hawkes_me=hawkes_main_effects, hawkes_int=hawkes_interactions, 
                D=D, verbose=False)

np.random.seed(117711)
## Prior initialisation
m.prior_initialisation()
## Initialise
init_i = np.array([Counter(m.ni)[x] / m.T / m.n1 + 1e-9 for x in range(m.n1)]) 
init_j = np.array([Counter(m.nj_prime)[x] / m.T / m.n2 + 1e-9 for x in range(m.n2)])
m.alpha = init_i
m.beta = init_j
if not poisson_main_effects:
    m.mu = init_i 
    m.mu_prime = init_j
    m.phi = 3 * init_i
    m.phi_prime = 3 * init_j

if m.D == 1:
    m.gamma = np.sqrt(init_i)
    m.gamma_prime = np.sqrt(init_j)
    if not poisson_interactions:    
        m.nu = np.repeat(1e-4, m.n1)
        m.nu_prime = np.repeat(1e-4, m.n2)
        m.theta = np.repeat(5 * 1e-4, m.n1)
        m.theta_prime = np.repeat(5 * 1e-4, m.n2)
else:
    for d in range(m.D):
        m.gamma[:,d] = np.repeat(1e-5, m.n1) + np.random.normal(scale=2e-6,size=m.n1)
        m.gamma_prime[:,d] = np.repeat(1e-5, m.n2) + np.random.normal(scale=2e-6,size=m.n2)
        if not poisson_interactions:  
            m.nu[:,d] = np.repeat(1e-5, m.n1) + np.random.normal(scale=2e-6,size=m.n1)
            m.nu_prime[:,d] = np.repeat(1e-5, m.n2) + np.random.normal(scale=2e-6,size=m.n2)
            m.theta[:,d] = np.repeat(5 * 1e-5, m.n1) + np.random.normal(scale=2e-6,size=m.n1)
            m.theta_prime[:,d] = np.repeat(5 * 1e-5, m.n2) + np.random.normal(scale=2e-6,size=m.n2)

## Likelihood optimisation
l = m.optimise_meg(prior_penalisation=False, learning_rate=0.5, method='adam', max_iter=250, verbose=False, tolerance=1e-5)

## Calculate the p-values
m.pvalues(A_test=G_test)
pp = [p for x in m.pvals_train.values() for p in list(x)]
pp_test = [p for x in m.pvals_test.values() for p in list(x)]

## Calculate the KS scores
from scipy import stats
ks_scores = np.zeros(2)
ks_scores[0] = stats.kstest(pp, 'uniform')[0]
ks_scores[1] = stats.kstest(pp_test, 'uniform')[0]

## Save files
np.savetxt(dest_folder + '/loglik.csv', l, fmt='%.20f', delimiter=',')
np.savetxt(dest_folder + '/pvals.csv', pp, fmt='%.20f', delimiter=',')
np.savetxt(dest_folder + '/pvals_test.csv', pp_test, fmt='%.20f', delimiter=',')
np.savetxt(dest_folder + '/ks_scores.csv', ks_scores, fmt='%.20f', delimiter=',')
if main_effects:
    np.savetxt(dest_folder + '/alpha.csv', m.alpha, fmt='%.20f', delimiter=',')
    np.savetxt(dest_folder + '/beta.csv', m.beta, fmt='%.20f', delimiter=',')
    if not poisson_main_effects:
        np.savetxt(dest_folder + '/mu.csv', m.mu, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/phi.csv', m.phi, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/mu_prime.csv', m.mu_prime, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/phi_prime.csv', m.phi_prime, fmt='%.20f', delimiter=',')
if interactions:
    np.savetxt(dest_folder + '/gamma.csv', m.gamma, fmt='%.20f', delimiter=',')
    np.savetxt(dest_folder + '/gamma_prime.csv', m.gamma_prime, fmt='%.20f', delimiter=',')
    if not poisson_interactions:
        np.savetxt(dest_folder + '/nu.csv', m.nu, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/theta.csv', m.theta, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/nu_prime.csv', m.nu_prime, fmt='%.20f', delimiter=',')
        np.savetxt(dest_folder + '/theta_prime.csv', m.theta_prime, fmt='%.20f', delimiter=',')
