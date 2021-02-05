#!/usr/bin/env python3
import sys
import numpy as np
from scipy.optimize import minimize
import math
import copy
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.special import expit
from sparse import COO
import scipy.stats
import numbers
import warnings
warnings.filterwarnings("error")

####################################
### Model from Fox et. al (2016) ###
####################################

class fox:
	
	## Initialise the class from the dynamic adjacency matrix A (or biadjacency matrix), stored in a dictionary
	def __init__(self,A, T=0, force_square=True, verbose=False, remove_self_loops=False):
		# Obtain total time of observation
		self.A = {}
		# Set the time of observation (assuming that the initial observation time is t=0)
		if verbose:
			print("+++ Creating the model object, sorting the events and calculating the total time of observation +++")
		if T == 0:
			self.T = 0
			A_del = {}
			for link in A:
				if len(A[link]) > 0:
					## Sorts the events and update the total time of observation
					self.A[link] = np.sort(A[link])
					q = np.max(self.A[link])
					if q > self.T:
						self.T = q
				else:
					A_del[link] = []
		else: 
			self.T = T
		# Remove redundant links (empty time series)
		if verbose:
			print("+++ Removing redundant links +++")
		for link in A_del:
			del self.A[link]
		# Obtain dimension of the matrix and type of graph (undirected, directed, bipartite)
		if verbose:
			print("+++ Calculating the dimension of the matrix and its characteristics (undirected, directed, bipartite) +++")		
		max_source = 0
		max_dest = 0
		for link in self.A:
			if link[0] > max_source:
				max_source = link[0]
			if link[1] > max_dest:
				max_dest = link[1]
		# If max_source != max_dest, then the graph is probably bipartite (force_square enforces non-bipartite graph)
		self.n = np.max([max_source,max_dest]) + 1
		# Total number of observations on each edge
		if verbose:
			print("+++ Calculating the total number of observations on each edge +++")
		self.nij = Counter()
		for link in self.A:
			self.nij[link] = len(self.A[link])
		self.m = np.sum(list(self.nij.values()))
		## Obtain sequences of source/destination nodes for each node
		if verbose:
			print("+++ Obtaining time series of node connections for each node +++")
		self.out_nodes = {}
		self.in_nodes = {}
		for link in self.A:
			if remove_self_loops:
				if link[0] != link[1]:
					if link[0] in self.out_nodes:
						self.out_nodes[link[0]] += [link[1]]
					else:
						self.out_nodes[link[0]] = [link[1]]
					if link[1] in self.in_nodes:
						self.in_nodes[link[1]] += [link[0]]
					else:
						self.in_nodes[link[1]] = [link[0]]
			else:
				if link[0] in self.out_nodes:
					self.out_nodes[link[0]] += [link[1]]
				else:
					self.out_nodes[link[0]] = [link[1]]
				if link[1] in self.in_nodes:
					self.in_nodes[link[1]] += [link[0]]
				else:
					self.in_nodes[link[1]] = [link[0]]

	## Model specification for node-based model
	def specification(self, verbose=False, remove_repeated_times=False, use_receiver=True):
		self.use_receiver = use_receiver
		## Parameters
		self.nu = np.zeros(self.n)
		self.omega = np.zeros(self.n)
		self.theta = np.zeros(self.n)
		## Objects containing the time series on each node
		self.node_ts_source = {}
		self.ni_source = {}
		n = len(self.out_nodes)
		if verbose:
				print("")
				prop = 0
		for node in self.out_nodes:
			if verbose:
				prop += 1
				print("\r+++ Percentage of processed destination nodes +++ {:0.2f}%".format(prop / n * 100), end="")
			self.node_ts_source[node] = []
			for dest_node in self.out_nodes[node]:
				self.node_ts_source[node] += list(self.A[node,dest_node])
			sort_node = np.argsort(self.node_ts_source[node])
			if remove_repeated_times:
				self.node_ts_source[node] = np.unique(np.array(self.node_ts_source[node])[sort_node])  
			else:
				self.node_ts_source[node] = np.array(self.node_ts_source[node])[sort_node]
			self.ni_source[node] = len(self.node_ts_source[node])
		## Repeat for the receivers
		if verbose:
				print("")
				prop = 0
		self.node_ts_receiver = {}
		self.ni_receiver = {}
		n = len(self.in_nodes)
		for node in self.in_nodes:
			if verbose:
				prop += 1
				print("\r+++ Percentage of processed source nodes +++ {:0.2f}%".format(prop / n * 100), end="")
			self.node_ts_receiver[node] = []
			for source_node in self.in_nodes[node]:
				self.node_ts_receiver[node] += list(self.A[source_node,node])
			sort_node = np.argsort(self.node_ts_receiver[node])
			if remove_repeated_times:
				self.node_ts_receiver[node] = np.unique(np.array(self.node_ts_receiver[node])[sort_node])
			else:
				self.node_ts_receiver[node] = np.array(self.node_ts_receiver[node])[sort_node]
			self.ni_receiver[node] = len(self.node_ts_receiver[node])
		if verbose:
			print("")
		## Combine source and receivers
		self.source_receiver = {}
		for node in self.out_nodes:
			self.source_receiver[node] = {}
			k = 0
			for time in self.node_ts_source[node]:
				if node in self.node_ts_receiver:
					if self.use_receiver:
						self.source_receiver[node][k] = time - self.node_ts_receiver[node][self.node_ts_receiver[node] < time]
					else:
						self.source_receiver[node][k] = time - self.node_ts_receiver[node][self.node_ts_receiver[node] < time]
				else: 
					self.source_receiver[node][k] = np.array([])
				k += 1

	## Compensator
	def compensator(self):
		self.Lambda = {}
		for node in self.node_ts_source:
			self.Lambda[node] = self.nu[node] * self.node_ts_source[node] 
			self.Lambda[node] += self.theta[node] * np.array([np.sum(1 - np.exp(-self.omega[node] * self.source_receiver[node][k])) for k in range(len(self.node_ts_source[node]))])

	## Calculate p-values on training and test set
	def pvalues(self, verbose=False):
		## Obtain the compensator values
		self.compensator()
		## Calculate the p-values
		self.pvals = {}
		for node in self.node_ts_source:
			self.pvals[node] = np.zeros(self.ni_source[node])
			self.pvals[node][0] = 1 - np.exp(-self.Lambda[node][0])
			self.pvals[node][1:] = 1 - np.exp(-np.diff(self.Lambda[node]))

## Calculate negative log-likelihood
def negative_loglikelihood(params, node, self):
	nu = np.exp(params[0])
	omega = np.exp(params[1])
	theta = expit(params[2])
	## Calculate the negative log-likelihood
	loglik = nu * self.T
	loglik += theta * np.sum(1 - np.exp(-omega * (self.T - self.node_ts_receiver[node] if self.use_receiver else self.node_ts_source[node])))
	loglik -= np.sum(np.log(nu + theta * omega * np.array([np.sum(np.exp(-omega * self.source_receiver[node][k])) for k in range(self.ni_source[node])])))
	return loglik 

def optimize_model(self, init_values):
	if (len(init_values[0]) != 3):
		raise ValueError('Incorrect dimension of the initial values.')
	self.nu = np.zeros(self.n)
	self.omega = np.zeros(self.n)
	self.theta = np.zeros(self.n)
	success_vector = -np.ones(self.n)
	q = {}
	k = 1
	for node in self.node_ts_source:
		print("\r+++ Percentage of processed source nodes +++ {:0.2f}%".format(k / len(self.node_ts_source)* 100), end="")
		k += 1
		try: 
			q[node] = minimize(negative_loglikelihood, x0=init_values[node], args=(node, self), method='Nelder-Mead')
		except:
			continue
	for node in range(self.n):
		try:
			success_vector[node] = q[node].success 
			res = q[node].x
			self.nu[node] = np.exp(res[0])
			self.omega[node] = np.exp(res[1])
			self.theta[node] = expit(res[2])
		except:
			continue
	print("")
	return success_vector, self.nu, self.omega, self.theta
