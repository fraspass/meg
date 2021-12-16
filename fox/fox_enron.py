#! /usr/bin/env python3
import fox_model
import os, argparse
from collections import Counter
from scipy.optimize import minimize
from scipy.special import logit
import numpy as np
from scipy.sparse import coo_matrix

## Add test set
test = True
G = {}
if test:
    G_test = {}

## Construct the directed graph
with open('../data/enron_tuples_filter.txt') as f:
    for line in f:
        line = line.rstrip('\r\n').split(',')
        time = int(line[0]) - 910948020
        link = tuple([int(x) for x in line[1:]])
        sour = link[0]
        dest = link[1]
        ## Add link to the graph
        if test:
            if time > (1007164800 - 910948020): 
                if link in G_test:
                    G_test[link] += [time]
                else:
                    G_test[link] = [time]                  
            else:
                if link in G:
                    G[link] += [time]
                else:
                    G[link] = [time]          
        else:
            if link in G:
                G[link] += [time]
            else:
                G[link] = [time]

## Arrival times
times = []
for link in G:
    times += G[link]

## Test set arrival times
if test:
    times_test = []
    for link in G_test:
        times_test += G_test[link]

## Sort times
times = np.sort(times)
if test:
    times_test = np.sort(times_test)

## Build the MEG model
m = fox_model.fox(G, verbose=False, remove_self_loops=True)
m.specification(verbose=True, use_receiver=True, remove_repeated_times=True)

## Initialisation
init_values = {}
for node in m.node_ts_source:
	init_values[node] = np.array([np.log(m.ni_source[node] / m.T / m.n), np.log(1e-4), logit(0.5)])

## Initial values must be on log-scale
q = fox_model.optimize_model(m, init_values=init_values)
m.nu = q[1]
m.omega = q[2]
m.theta = q[3] 
m.pvalues()

## KS score
pp = [p for x in m.pvals.values() for p in list(x)]
from scipy import stats
ks_scores = stats.kstest(pp, 'uniform')
print('KS score (Hawkes receivers model): ', ks_scores[0])

## Poisson p-values
poisson_pvals = {}
for node in m.node_ts_source:
	nu_hat = m.ni_source[node] / m.T
	poisson_pvals[node] = 1 - np.exp(- nu_hat * np.hstack([m.node_ts_source[node][0],np.diff(m.node_ts_source[node])]))

## Poisson KS score
pp_poisson = [p for x in poisson_pvals.values() for p in list(x)]
ks_poisson = stats.kstest(pp_poisson, 'uniform')
print('KS score (Poisson model): ', ks_poisson[0])

## Repeat for the receivers
m.specification(verbose=True, use_receiver=False, remove_repeated_times=True)
q = fox_model.optimize_model(m, init_values=init_values)
m.nu = q[1]
m.omega = q[2]
m.theta = q[3] 
m.pvalues()

## KS score for the model for each source, based on source sending times
pp = [p for x in m.pvals.values() for p in list(x)]
ks_scores2 = stats.kstest(pp, 'uniform')
print('KS score (Hawkes sources model): ', ks_scores2[0])