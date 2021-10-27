#!/usr/bin/env python3
import sys
import numpy as np
from numpy import pi,log,exp,sqrt
import math
import copy
from collections import Counter
from scipy.sparse import coo_matrix
from sparse import COO
import scipy.stats
import numbers
import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("error")

## Scaled exponential excitation functions

# Function for the main effects
def scaled_exponential(t,beta,theta):
	return beta * exp(- (beta + theta) * t)

# Function for the interactions
def scaled_exponential_prod(t,beta1,beta2,theta1,theta2):
	return np.sum(beta1 * beta2 * exp(- (beta1 + theta1) * (beta2 + theta2) * t))

## Vectorize the operation
## scaled_exponential_prod_vec = np.vectorize(scaled_exponential_prod)
def scaled_exponential_prod_vec(t,beta1,beta2,theta1,theta2):
		out = np.zeros(len(t))
		for i in range(len(t)):
			out[i] = scaled_exponential_prod(t[i],beta1,beta2,theta1,theta2)
		return out

# Difference for discrete process (almost equivalent to np.diff)
def discrete_process_difference(ts):
	diff = []
	start = np.sum(np.array(ts) == ts[0])
	index = copy.copy(start)
	for time in ts[start:]:
		while ts[index] < time:
			index += 1
		diff += [time - ts[index - 1]]
	return np.array(diff)

###################################################
### Mutually exciting point process graph model ###
###################################################

class meg_model:
	
	## Initialise the class from the dynamic adjacency matrix A (or biadjacency matrix), stored in a dictionary
	def __init__(self,A, T=0, tau_zero=False, full_links=False, force_square=False, discrete=False, verbose=False, evaluate_directed=True):
		# Obtain total time of observation
		self.A = {} #A
		# Discrete/continuous model
		if discrete:
			self.discrete = True
		else:
			self.discrete = False
		# Set the time of observation (assuming that the initial observation time is t=0)
		if verbose:
			print("+++ Creating the MEG model object, sorting the events and calculating the total time of observation +++")
		if T == 0:
			self.T = 0
			for link in A:
				if len(A[link]) > 0:
					## Sorts the events and update the total time of observation
					self.A[link] = np.sort(A[link])
					q = np.max(self.A[link])
					if q > self.T:
						self.T = q
		else: 
			self.T = T
			for link in A:
				if len(np.array(A[link])[np.array(A[link]) < self.T]) > 0:
					self.A[link] = np.sort(np.array(A[link])[np.array(A[link]) < self.T])
		# Remove redundant links (empty time series)
		if verbose:
			print("+++ Removing redundant links +++")
		# Obtain dimension of the matrix and type of graph (undirected, directed, bipartite)
		if verbose:
			print("+++ Calculating the dimension of the matrix and its characteristics (undirected, directed, bipartite) +++")		
		count_symmetries = 0
		max_source = 0
		max_dest = 0
		for link in self.A:
			if link[0] > max_source:
				max_source = link[0]
			if link[1] > max_dest:
				max_dest = link[1]
			if link[::-1] in self.A: # and len(self.A[link]) == len(self.A[link[::-1]]) and np.all(self.A[link] == self.A[link[::-1]]):
				count_symmetries += 1
		# If max_source != max_dest, then the graph is probably bipartite (force_square enforces non-bipartite graph)
		if max_source != max_dest and not force_square:
			self.bipartite = True
			self.directed = True
			self.n1 = max_source + 1
			self.n2 = max_dest + 1
		else:
			self.bipartite = False
			self.n = np.max([max_source,max_dest]) + 1
			if (count_symmetries == len(self.A) or count_symmetries == 0) and evaluate_directed:
				self.directed = False
			else:
				self.directed = True
		## If the graph is undirected, remove unnecessary (redundant) edges (upper triangular representation of the dynamic adjacency matrix)
		if not self.directed:
			del_links = {}
			for link in self.A:
				if link[0] < link[1] and link[::-1] in self.A:
					del_links[link[::-1]] = 0
			for link in del_links:
				del self.A[link]
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
		if self.directed:
			self.in_nodes = {}
		for link in self.A:
			if link[0] in self.out_nodes:
				self.out_nodes[link[0]] += [link[1]]
			else:
				self.out_nodes[link[0]] = [link[1]]
			if not self.directed:
				if link[1] in self.out_nodes:
					self.out_nodes[link[1]] += [link[0]]
				else:
					self.out_nodes[link[1]] = [link[0]]
			else:
				if link[1] in self.in_nodes:
					self.in_nodes[link[1]] += [link[0]]
				else:
					self.in_nodes[link[1]] = [link[0]]
		## Define Tau
		self.full_links = full_links
		self.tau_zero = tau_zero
		if self.full_links and not self.tau_zero:
			return ValueError('If full_links is True, then tau_zero must be set to True.')
		self.Tau = {}
		for link in self.A:
			self.Tau[link] = 0 if tau_zero else self.A[link][0]

	## Model specification
	def specification(self, main_effects=True, interactions=True, poisson_me=False, poisson_int=False, hawkes_me=True, hawkes_int=True, D=1, verbose=False):
		prop = 0
		## Initialise the number of parameters and number of latent features
		self.n_parameters = 0
		self.D = D
		## Only Poisson process
		if poisson_me:
			self.poisson_me = True
		else:
			self.poisson_me = False
		if poisson_int:
			self.poisson_int = True
		else:
			self.poisson_int = False
		## Main effects: include dependence on events on each source or destination
		if main_effects == True:
			self.main_effects = main_effects
			if self.bipartite:
				self.n_parameters += (1 if self.poisson_me else 3) * (self.n1 + self.n2)
			else:
				self.n_parameters += (6 if self.directed else 3) / (3 if self.poisson_me else 1) * self.n
			## Use Hawkes process for the main effects, or a first order dependence only
			if hawkes_me:
				self.hawkes_me = hawkes_me
			else:
				self.hawkes_me = False
		else:
			self.main_effects = False
		## Objects containing the time series on each node
		## If the dictionary dict is indexed by node, then for undirected graph there is no dict_prime version, otherwise it exists
		## If the dictionary is indexed by edge, then the 'prime' version exists
		self.node_ts = {}
		self.node_ts_edges = {}
		self.node_ts_start = {}
		self.ni = {}
		n = len(self.out_nodes)
		for node in self.out_nodes:
			if verbose:
				prop += 1
				print("\r+++ Percentage of processed nodes +++ {:0.2f}%".format(prop / n * 100), end="")
			self.node_ts[node] = []
			self.node_ts_edges[node] = []
			for dest_node in self.out_nodes[node]:
				if self.directed:
					self.node_ts[node] += list(self.A[node,dest_node])
					self.node_ts_edges[node] += [dest_node] * len(self.A[node,dest_node])
				else:
					if (node,dest_node) in self.A:
						self.node_ts[node] += list(self.A[node,dest_node])
						self.node_ts_edges[node] += [dest_node] * len(self.A[node,dest_node])
					else:
						self.node_ts[node] += list(self.A[dest_node,node])
						self.node_ts_edges[node] += [dest_node] * len(self.A[dest_node,node])
			sort_node = np.argsort(self.node_ts[node])
			self.node_ts[node] = np.array(self.node_ts[node])[sort_node]
			self.ni[node] = len(sort_node)
			## node_ts_edges contains the time series of destination nodes corresponding to the given source (to match with arrival times)
			self.node_ts_edges[node] = np.array(self.node_ts_edges[node])[sort_node]
			## node_ts_start contains the first destination node for each source (or sequence of destination nodes if there are ties)
			q = np.sum(self.node_ts[node] == self.node_ts[node][0]) if self.discrete else 1
			self.node_ts_start[node] = self.node_ts_edges[node][:q] #[0]
		## Repeat for the destinations if the graph is directed
		if self.directed:	
			if verbose:
				print("")
				prop = 0
			self.node_ts_prime = {}
			self.node_ts_prime_edges = {}
			self.node_ts_prime_start = {}
			self.nj_prime = {}
			n = len(self.in_nodes)
			for node in self.in_nodes:
				if verbose:
					prop += 1
					print("\r+++ Percentage of processed nodes (directed graphs) +++ {:0.2f}%".format(prop / n * 100), end="")
				self.node_ts_prime[node] = []
				self.node_ts_prime_edges[node] = []
				for source_node in self.in_nodes[node]:
					self.node_ts_prime[node] += list(self.A[source_node,node])
					self.node_ts_prime_edges[node] += [source_node]*len(self.A[source_node,node])
				sort_node = np.argsort(self.node_ts_prime[node])
				self.node_ts_prime[node] = np.array(self.node_ts_prime[node])[sort_node]
				self.nj_prime[node] = len(sort_node)
				self.node_ts_prime_edges[node] = np.array(self.node_ts_prime_edges[node])[sort_node]
				q = np.sum(self.node_ts_prime[node] == self.node_ts_prime[node][0]) if self.discrete else 1
				self.node_ts_prime_start[node] = self.node_ts_prime_edges[node][:q] #[0]
			if verbose:
				print("")
		## Interactions: include dependence on events on each edge
		if interactions == True:
			self.interactions = interactions
			## Use Hawkes process for the interactions, or a first order dependence only
			if hawkes_int == True:
				self.hawkes_int = hawkes_int
				if self.bipartite:	
					self.n_parameters += self.D * (1 if self.poisson_int else 3) * (self.n1+self.n2)
				else:
					self.n_parameters += self.D * (6 if self.directed else 3) / (3 if self.poisson_int else 1) * self.n
			else:
				self.hawkes_int = False
		else:
			self.interactions = False
		## If neither main effects or interactions are specified, return an error
		if not self.main_effects and not self.interactions:
			return ValueError('Must specify at least the main effects or interactions.')

	## Parameter initialisation from the prior distribution (not recommended)
	def prior_initialisation(self,alpha=[1,10],beta=[1,10],phi=[1,10],phi_prime=[1,10],mu=[1,10],mu_prime=[1,10],
				gamma=[1,10],gamma_prime=[1,10],theta=[1,10],theta_prime=[1,10],nu=[1,10],nu_prime=[1,10]):
		if np.all([type(x) is list and len(x)==2 for x in [alpha,beta,phi,mu,mu_prime,gamma,gamma_prime,theta,theta_prime,nu,nu_prime]]):	
			## Initialise the main effects hyperparameters
			if self.main_effects:
				self.hyper_alpha = alpha
				if self.directed:
					self.hyper_beta = beta
				if not self.poisson_me:
					self.hyper_phi = phi
					if self.directed:
						self.hyper_phi_prime = phi_prime
					self.hyper_mu = mu
					if self.directed:
						self.hyper_mu_prime = mu_prime
				## Sample from the gamma distributions (numbers depending on the type of graph)
				## Alpha
				self.tau = np.random.gamma(shape=self.hyper_alpha[0],scale=1/self.hyper_alpha[1],size=2)
				self.alpha = np.random.gamma(shape=self.tau[0],scale=1/self.tau[1],size=self.n1 if self.bipartite else self.n)
				## Beta
				if self.directed:
					self.tau_prime = np.random.gamma(shape=self.hyper_beta[0],scale=1/self.hyper_beta[1],size=2)
					self.beta = np.random.gamma(shape=self.tau_prime[0],scale=1/self.tau_prime[1],size=self.n2 if self.bipartite else self.n)
				if not self.poisson_me:
					## Mu
					self.csi = np.random.gamma(shape=self.hyper_mu[0],scale=1/self.hyper_mu[1],size=2)
					self.mu = np.random.gamma(shape=self.csi[0],scale=1/self.csi[1],size=self.n1 if self.bipartite else self.n)
					if self.directed:
						self.csi_prime = np.random.gamma(shape=self.hyper_mu_prime[0],scale=1/self.hyper_mu_prime[1],size=2)
						self.mu_prime = np.random.gamma(shape=self.csi_prime[0],scale=1/self.csi_prime[1],size=self.n2 if self.bipartite else self.n)
					## Phi
					self.kappa = np.random.gamma(shape=self.hyper_phi[0],scale=1/self.hyper_phi[1],size=2)
					self.phi = np.random.gamma(shape=self.kappa[0],scale=1/self.kappa[1],size=self.n1 if self.bipartite else self.n)
					if self.directed:
						self.kappa_prime = np.random.gamma(shape=self.hyper_phi_prime[0],scale=1/self.hyper_phi_prime[1],size=2)
						self.phi_prime = np.random.gamma(shape=self.kappa_prime[0],scale=1/self.kappa_prime[1],size=self.n2 if self.bipartite else self.n)
			## Initialise the interactions hyperparameters
			if self.interactions:
				self.hyper_gamma = gamma
				if self.directed:
					self.hyper_gamma_prime = gamma_prime
				if not self.poisson_int:
					self.hyper_theta = theta
					if self.directed:
						self.hyper_theta_prime = theta_prime
					self.hyper_nu = nu
					if self.directed:
						self.hyper_nu_prime = nu_prime
				## Sample from the gamma distributions (numbers depending on the type of graph)
				## Gamma
				self.tau_tilde = np.random.gamma(shape=self.hyper_gamma[0],scale=1/self.hyper_gamma[1],size=2)
				if self.D > 1:
					self.gamma = np.random.gamma(shape=self.tau_tilde[0],scale=1/self.tau_tilde[1],size=(self.n1 if self.bipartite else self.n,self.D))
				else:
					self.gamma = np.random.gamma(shape=self.tau_tilde[0],scale=1/self.tau_tilde[1],size=(self.n1 if self.bipartite else self.n))
				if self.directed:
					self.tau_tilde_prime = np.random.gamma(shape=self.hyper_gamma_prime[0],scale=1/self.hyper_gamma_prime[1],size=2)
					if self.D > 1:
						self.gamma_prime = np.random.gamma(shape=self.tau_tilde_prime[0],scale=1/self.tau_tilde_prime[1],size=(self.n2 if self.bipartite else self.n,self.D))
					else:
						self.gamma_prime = np.random.gamma(shape=self.tau_tilde_prime[0],scale=1/self.tau_tilde_prime[1],size=(self.n2 if self.bipartite else self.n))
				if not self.poisson_int:
					## Nu
					self.csi_tilde = np.random.gamma(shape=self.hyper_nu[0],scale=1/self.hyper_nu[1],size=2)
					if self.D > 1:
						self.nu = np.random.gamma(shape=self.csi_tilde[0],scale=1/self.csi_tilde[1],size=(self.n1 if self.bipartite else self.n,self.D))
					else:
						self.nu = np.random.gamma(shape=self.csi_tilde[0],scale=1/self.csi_tilde[1],size=(self.n1 if self.bipartite else self.n))
					if self.directed:
						self.csi_tilde_prime = np.random.gamma(shape=self.hyper_nu_prime[0],scale=1/self.hyper_nu_prime[1],size=2)
						if self.D > 1:
							self.nu_prime = np.random.gamma(shape=self.csi_tilde_prime[0],scale=1/self.csi_tilde_prime[1],size=(self.n2 if self.bipartite else self.n,self.D))
						else:
							self.nu_prime = np.random.gamma(shape=self.csi_tilde_prime[0],scale=1/self.csi_tilde_prime[1],size=(self.n2 if self.bipartite else self.n))
					## Theta
					self.kappa_tilde = np.random.gamma(shape=self.hyper_theta[0],scale=1/self.hyper_theta[1],size=2)
					if self.D > 1:
						self.theta = np.random.gamma(shape=self.kappa_tilde[0],scale=1/self.kappa_tilde[1],size=(self.n1 if self.bipartite else self.n,self.D))
					else:
						self.theta = np.random.gamma(shape=self.kappa_tilde[0],scale=1/self.kappa_tilde[1],size=(self.n1 if self.bipartite else self.n))
					if self.directed:
						self.kappa_tilde_prime = np.random.gamma(shape=self.hyper_theta_prime[0],scale=1/self.hyper_theta_prime[1],size=2)
						if self.D > 1:	
							self.theta_prime = np.random.gamma(shape=self.kappa_tilde_prime[0],scale=1/self.kappa_tilde_prime[1],size=(self.n2 if self.bipartite else self.n,self.D))
						else:
							self.theta_prime = np.random.gamma(shape=self.kappa_tilde_prime[0],scale=1/self.kappa_tilde_prime[1],size=(self.n2 if self.bipartite else self.n))	
		else:
			return ValueError('Hyperparameters must be two dimensional vectors.')

	## Calculation of the intensity for a given time t (mostly useful for the simulation of the process)
	def intensity(self,t,right_limit=False):
		## If t is negative, returns an error
		if t < 0:
			return ValueError('t must be non-negative')
		## Initialise dictionary of intensitites (counter object)
		lambda_ij = Counter()
		for link in self.A:
			## Check if there are events on each node which occurred before the time t
			t_cond = (self.node_ts[link[0]] <= t) if right_limit else (self.node_ts[link[0]] < t)
			after_first_event = np.any(t_cond)
			if after_first_event:
				max_ind = np.max(np.where(t_cond))
			if self.directed:
				t_cond_prime = (self.node_ts_prime[link[1]] <= t) if right_limit else (self.node_ts_prime[link[1]] < t)
				after_first_event_prime = np.any(t_cond_prime)
				if after_first_event_prime:
					max_ind_prime = np.max(np.where(t_cond_prime))
			else:
				t_cond_prime = (self.node_ts[link[1]] <= t) if right_limit else (self.node_ts[link[1]] < t)
				after_first_event_prime = np.any(t_cond_prime)
				if after_first_event_prime:
					max_ind_prime = np.max(np.where(t_cond_prime))
			## Add intensities for main effects (when present)
			if self.main_effects:
				lambda_ij[link] += self.alpha[link[0]]
				lambda_ij[link] += self.beta[link[1]] if self.directed else self.alpha[link[1]]
				if not self.poisson_me:
					if self.hawkes_me:
						lambda_ij[link] += 0 if not after_first_event else np.sum(scaled_exponential(t-self.node_ts[link[0]][:(max_ind+1)], self.mu[link[0]],self.phi[link[0]]))
						lambda_ij[link] += 0 if not after_first_event_prime else np.sum(scaled_exponential(t-(self.node_ts_prime[link[1]][:(max_ind_prime+1)] if self.directed else self.node_ts[link[1]][:(max_ind_prime+1)]),
															self.mu_prime[link[1]] if self.directed else self.mu[link[1]], self.phi_prime[link[1]] if self.directed else self.phi[link[1]]))
					else:
						lambda_ij[link] += 0 if not after_first_event else scaled_exponential(t-self.node_ts[link[0]][max_ind], self.mu[link[0]],self.phi[link[0]])
						lambda_ij[link] +=  0 if not after_first_event_prime else scaled_exponential(t-(self.node_ts_prime[link[1]][max_ind_prime] if self.directed else self.node_ts[link[1]][max_ind_prime]),
															self.mu_prime[link[1]] if self.directed else self.mu[link[1]], self.phi_prime[link[1]] if self.directed else self.phi[link[1]])
			## Add intensities for interactions (when present)
			if self.interactions:
				lambda_ij[link] += np.sum(self.gamma[link[0]] * self.gamma_prime[link[1]])
				if not self.poisson_int:
					after_first_edge_event = np.any(np.array(self.A[link]) < t)
					if after_first_edge_event:
						max_edge_ind = np.max(np.where(np.array(self.A[link]) < t))
					if self.hawkes_int:
						if not after_first_edge_event:
							lambda_ij[link] += 0
						else:
							if len(np.array(self.A[link])[:(max_edge_ind+1)]) <= 1:
								lambda_ij[link] += np.sum(scaled_exponential_prod(t-np.array(self.A[link])[:(max_edge_ind+1)],
											self.nu[link[0]], self.nu_prime[link[1]] if self.directed else self.nu[link[1]], self.theta[link[0]], self.theta_prime[link[1]] if self.directed else self.theta[link[1]]))
							else:
								lambda_ij[link] += np.sum(scaled_exponential_prod_vec(t-np.array(self.A[link])[:(max_edge_ind+1)],
											self.nu[link[0]], self.nu_prime[link[1]] if self.directed else self.nu[link[1]], self.theta[link[0]], self.theta_prime[link[1]] if self.directed else self.theta[link[1]]))
					else:
						lambda_ij[link] += 0 if not after_first_edge_event else np.sum(scaled_exponential_prod(t-self.A[link][max_edge_ind],
										self.nu[link[0]], self.nu_prime[link[1]] if self.directed else self.nu[link[1]], self.theta[link[0]], self.theta_prime[link[1]] if self.directed else self.theta[link[1]]))
		## Sum to obtain overall intensity
		lambda_tot = sum(lambda_ij.values())
		## Return the outputs
		return lambda_ij, lambda_tot

	## Simulation of the process
	def simulate(self, T=None, m=None, copy_dict=False, verbose=False):
		if T is None:
			T = self.T
		if m is None:
			m = 1e9
		## Initialise adjacency matrix
		if copy_dict:
			self.A_initial = copy.deepcopy(self.A)
		## Dynamic adjacency matrix must be empty
		for link in self.A:
			self.A[link] = []
			self.node_ts[link[0]] = np. array([])
			self.node_ts_prime[link[1]] = np. array([])
		## Initialise arrival times
		t_star = 0
		n_events = 0
		while t_star < T:
			## If the graph is discrete, include the current arrival time
			_, lambda_star = self.intensity(t_star, right_limit=self.discrete)
			## Propose new arrival time
			if self.discrete:
				t_star += math.floor(-math.log(np.random.uniform(size=1)) / lambda_star)
			else:
				t_star -= math.log(np.random.uniform(size=1)) / lambda_star
			if t_star > T or n_events >= m:
				if verbose:
					print("")
				break
			## Calculate intensities for each edge
			lambda_ij, _ = self.intensity(t_star)
			## Calculate probabilities 
			links = list(lambda_ij.keys())
			probs = [x / lambda_star for x in list(lambda_ij.values())]
			self.probs = probs
			self.t_star = t_star
			sum_probs = np.sum(probs)
			if sum_probs < 1:
				probs += [1-sum_probs]
				links += ['Reject']
			else:
				probs = [x / sum_probs for x in probs]
			## Assign the arrival time to one of the links
			assignment = np.random.choice(len(links),p=probs)
			if links[assignment] != 'Reject':
				n_events += 1
				self.A[links[assignment]] = np.append(self.A[links[assignment]],t_star)
				self.node_ts[links[assignment][0]] = np.append(self.node_ts[links[assignment][0]], t_star)
				self.node_ts_prime[links[assignment][1]] += [t_star]
			if verbose:
				print("\r+++ Number of simulated events +++ {} - Time: {:0.3f}".format(n_events,t_star), end="")
		if verbose:
			print("")

	## Calculate required quantities for evaluation of the likelihood
	def likelihood_calculation_setup(self, verbose=False):
		## Calculate the inter-arrival times on each edge
		if verbose:
			prop = 0
		self.A_diff = {}
		if self.discrete:
			self.equal_start = Counter()
		for link in self.A:
			## if self.nij[link] > 1:	
			self.A_diff[link] = discrete_process_difference(self.A[link]) if self.discrete else np.diff(self.A[link])
			if self.discrete:
				self.equal_start[link] = np.sum(self.A[link] == self.A[link][0])
			if verbose:
				prop += self.nij[link]
				print("\r+++ Percentage of processed links (time differences) +++ {:0.2f}%".format(prop / self.m * 100), end="")	
		if verbose:
			prop = 0
			print("")
		if not self.poisson_me:
			if self.main_effects:
				## Set up the calculations for the psi coefficients (only for Hawkes process)
				if self.hawkes_me:
					self.psi_times = {}
					self.psi_prime_times = {}
					for link in self.A:
						## Initialise the dictionary of dictionaries for the time differences used to calculate each psi
						self.psi_times[link] = {}
						self.psi_prime_times[link] = {}
						## Obtain ell_k indices (connections on the given edge on the node time series)
						ell_k = np.where(self.node_ts_edges[link[0]] == link[1])[0]
						ts = self.node_ts[link[0]]
						if self.discrete:
							for k in range(len(ell_k)):
								arrival_time = ts[ell_k[k]]
								k_mod = ell_k[k]-1
								while ts[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k[k] = k_mod+1
						## if len(ell_k) != self.nij[link]:
							## raise ValueError('len(ell_k) and nij must be equivalent.')
						## Sequentially obtain the arrival times needed to calculate psi(k)
						for k in range(len(ell_k)):
							self.psi_times[link][k] = self.A[link][k] - ts[(ell_k[k-1]+1 if k != 0 else 0):ell_k[k]]
						## Repeat for psi_prime
						ell_k_prime = np.where((self.node_ts_prime_edges[link[1]] if self.directed else self.node_ts_edges[link[1]]) == link[0])[0]
						ts_prime = (self.node_ts_prime if self.directed else self.node_ts)[link[1]]
						if self.discrete:
							for k in range(len(ell_k_prime)):
								arrival_time = ts_prime[ell_k_prime[k]]
								k_mod = ell_k_prime[k]-1
								while ts_prime[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k_prime[k] = k_mod+1
						## if len(ell_k_prime) != self.nij[link]:
							## raise ValueError('len(ell_k_prime) and nij must be equivalent.')
						for k in range(len(ell_k_prime)):
							self.psi_prime_times[link][k] = self.A[link][k] - ts_prime[(ell_k_prime[k-1]+1 if k != 0 else 0):ell_k_prime[k]]
						if verbose:
							prop += self.nij[link]
							print("\r+++ Percentage of processed links (separation of arrival times) +++ {:0.2f}%".format(prop / self.m * 100), end="")
				else:
					## Set up the calculations for the t bar arrival times (only for r=1 process)
					self.A_bar = {}
					self.A_bar_prime = {}
					for link in self.A:
						## Initialise the t bars
						self.A_bar[link] = []
						self.A_bar_prime[link] = []
						## Obtain ell_k indices (connections on the given edge on the node time series)
						ell_k = np.where(self.node_ts_edges[link[0]] == link[1])[0] - 1
						ts = self.node_ts[link[0]]
						if self.discrete:
							for k in range(len(ell_k)):
								arrival_time = ts[ell_k[k]+1]
								k_mod = ell_k[k]
								while ts[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k[k] = k_mod
						indices = (ell_k >= 0)
						self.A_bar[link] = self.A[link][indices] - ts[ell_k[indices]]
						## Repeat for psi_prime
						ell_k_prime = np.where((self.node_ts_prime_edges[link[1]] if self.directed else self.node_ts_edges[link[1]]) == link[0])[0] - 1
						ts_prime = (self.node_ts_prime if self.directed else self.node_ts)[link[1]]
						if self.discrete:
							for k in range(len(ell_k_prime)):
								arrival_time = ts_prime[ell_k_prime[k]+1]
								k_mod = ell_k_prime[k]
								while ts_prime[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k_prime[k] = k_mod
						indices_prime = (ell_k_prime >= 0)
						self.A_bar_prime[link] = self.A[link][indices_prime] - ts_prime[ell_k_prime[indices_prime]]
						if verbose:
							prop += self.nij[link]
							print("\r+++ Percentage of processed links (calculation of t_bars) +++ {:0.2f}%".format(prop / self.m * 100), end="")
			if verbose:
				print("")

	## Recursive calculations of psi
	def psi_calculation(self, calculate_derivative=False, verbose=False):
		## Psi coefficients are only required when the Hawkes process is used
		# if (self.main_effects and (self.poisson_me or not self.hawkes_me)) or (self.interactions and (self.poisson_int or not self.hawkes_int)):
		# 	warnings.warn("psi_calculation should be used only if Hawkes processes for main effects or interactions are used.", Warning)
		if not self.poisson_me or not self.poisson_int:
			## Main effects with Hawkes 
			if self.main_effects and not self.poisson_me and self.hawkes_me:
				## Initialise the dictionaries
				self.psi = {}
				self.psi_prime = {}
				## Initialise the dictionaries for the derivatives (if required)
				if calculate_derivative:
					self.psi_derivative = {}
					self.psi_prime_derivative = {}
				if verbose:
					prop = 0
				for link in self.A:
					## Initialise the dictionary of dictionaries for each link (key 1: link, key 2: k)
					self.psi[link] = {}
					self.psi_prime[link] = {}
					if calculate_derivative:
						self.psi_derivative[link] = {}
						self.psi_prime_derivative[link] = {} 
					## Obtain the parameters
					mu = self.mu[link[0]]
					mu_prime = self.mu_prime[link[1]] if self.directed else self.mu[link[1]]
					phi = self.phi[link[0]]
					phi_prime = self.phi_prime[link[1]] if self.directed else self.phi[link[1]]
					## Obtain the times from psi_times
					times = self.psi_times[link][0]
					times_prime = self.psi_prime_times[link][0]
					## Calculate the required psi (and derivatives if required)
					self.psi[link][0] = np.sum(np.exp(-(mu+phi) * times))
					self.psi_prime[link][0] = np.sum(np.exp(-(mu_prime+phi_prime) * times_prime))
					if calculate_derivative:
						self.psi_derivative[link][0] = - np.sum(times * np.exp(-(mu+phi) * times))
						self.psi_prime_derivative[link][0] = - np.sum(times_prime * np.exp(-(mu_prime+phi_prime) * times_prime))
					## Calculate the remaining values of psi sequentially
					if self.nij[link] > 1:
						tdiff = np.diff(self.A[link]) if self.discrete else self.A_diff[link]
						for k in range(1,self.nij[link]):
							times = self.psi_times[link][k]
							times_prime = self.psi_prime_times[link][k]
							t_diff = tdiff[k-1]
							self.psi[link][k] = np.exp(-(mu+phi) * t_diff) * (1 + self.psi[link][k-1]) + (np.sum(np.exp(-(mu+phi) * times)) if len(times) > 0 else 0)
							self.psi_prime[link][k] = np.exp(-(mu_prime+phi_prime) * t_diff) * (1 + self.psi_prime[link][k-1]) + (np.sum(np.exp(-(mu_prime+phi_prime) * times_prime)) if len(times_prime) > 0 else 0)
							if calculate_derivative:
								self.psi_derivative[link][k] = np.exp(-(mu+phi) * t_diff) * (self.psi_derivative[link][k-1] - t_diff * (1 + self.psi[link][k-1])) - (np.sum(times * np.exp(-(mu+phi) * times)) if len(times) > 0 else 0)
								self.psi_prime_derivative[link][k] = np.exp(-(mu_prime+phi_prime) * t_diff) * (self.psi_prime_derivative[link][k-1] - t_diff * (1 + self.psi_prime[link][k-1])) - \
																	(np.sum(times_prime * np.exp(-(mu_prime+phi_prime) * times_prime)) if len(times_prime) > 0 else 0)
						## Adjust for repeated values (ties)
						if self.discrete:
							for k in np.where(tdiff == 0)[0]:
								self.psi[link][k+1] = self.psi[link][k]
								self.psi_prime[link][k+1] = self.psi_prime[link][k]
								if calculate_derivative:
									self.psi_derivative[link][k+1] = self.psi_derivative[link][k]
									self.psi_prime_derivative[link][k+1] = self.psi_prime_derivative[link][k]
					if verbose:
						prop += self.nij[link]
						print("\r+++ Percentage of processed links (main effects) +++ {:0.2f}%".format(prop / self.m * 100), end="")
				if verbose:
					print("")
			## Interactions with Hawkes
			if self.interactions and not self.poisson_int and self.hawkes_int:
				self.psi_tilde = {}
				if calculate_derivative:
					self.psi_tilde_derivative = {}
					self.psi_tilde_derivative_prime = {}
				if verbose:
					prop = 0		
				for link in self.A:
					## For k=0, psi_tilde is 0
					self.psi_tilde[link] = {}
					self.psi_tilde[link][0] = np.zeros(self.D) if self.D > 1 else 0.0
					if calculate_derivative:
						self.psi_tilde_derivative[link] = {}
						self.psi_tilde_derivative_prime[link] = {}
						self.psi_tilde_derivative[link][0] = np.zeros(self.D) if self.D > 1 else 0.0
						self.psi_tilde_derivative_prime[link][0] = np.zeros(self.D) if self.D > 1 else 0.0
					## Obtain the parameters
					nu = self.nu[link[0]]
					nu_prime = self.nu_prime[link[1]] if self.directed else self.nu[link[1]]
					theta = self.theta[link[0]]
					theta_prime = self.theta_prime[link[1]] if self.directed else self.theta[link[1]]
					## Calculate the remaining values of psi sequentially
					if self.nij[link] > 1:
						tdiff = np.diff(self.A[link]) if self.discrete else self.A_diff[link]
						for k in range(1,self.nij[link]):
							t_diff = tdiff[k-1]		
							self.psi_tilde[link][k] = np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff) * (1 + self.psi_tilde[link][k-1])
							if calculate_derivative:	
								self.psi_tilde_derivative[link][k] = np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff) * (self.psi_tilde_derivative[link][k-1] - (nu_prime+theta_prime) * t_diff * (1 + self.psi_tilde[link][k-1])) 
								self.psi_tilde_derivative_prime[link][k] = np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff) * (self.psi_tilde_derivative_prime[link][k-1] - (nu+theta) * t_diff * (1 + self.psi_tilde[link][k-1]))
						## Adjust for repeated values (ties)
						if self.discrete:
							for k in np.where(tdiff == 0)[0]:
								self.psi_tilde[link][k+1] = self.psi_tilde[link][k]
								if calculate_derivative:
									self.psi_tilde_derivative[link][k+1] = self.psi_tilde_derivative[link][k]
									self.psi_tilde_derivative_prime[link][k+1] = self.psi_tilde_derivative_prime[link][k]	
					if verbose:
						prop += self.nij[link]
						print("\r+++ Percentage of processed links (interactions) +++ {:0.2f}%".format(prop / self.m * 100), end="")
				if verbose:
					print("")	
	
	## Recursive calculations of zeta (corresponding to lambda_ij for the observed links)
	def zeta_calculation(self, verbose=False):
		## Initialise the dictionary of dictionaries (key 1: link, key 2: k)
		self.zeta = {}
		if verbose: 
			prop = 0
		for link in self.A:
			self.zeta[link] = {}
			if self.main_effects:
				alpha = self.alpha[link[0]]
				beta = self.beta[link[1]] if self.directed else self.alpha[link[1]]
				if not self.poisson_me:
					mu = self.mu[link[0]]
					mu_prime = self.mu_prime[link[1]] if self.directed else self.mu[link[1]]
					## Parameters needed if Hawkes processes are not used (otherwise psi should be available)
					if not self.hawkes_me:
						phi = self.phi[link[0]]
						phi_prime = self.phi_prime[link[1]] if self.directed else self.phi[link[1]]
			if self.interactions:
				gamma = self.gamma[link[0]]
				gamma_prime = self.gamma_prime[link[1]] if self.directed else self.gamma[link[1]]
				if not self.poisson_int:
					nu = self.nu[link[0]]
					nu_prime = self.nu_prime[link[1]] if self.directed else self.nu[link[1]]
					## Parameters needed if Hawkes processes are not used (otherwise psi should be available)
					if not self.hawkes_int:
						theta = self.theta[link[0]]
						theta_prime = self.theta_prime[link[1]] if self.directed else self.theta[link[1]]
			## Check if the first link recorded on the time series for each node corresponds to the current link (important for the indices in the next part)
			if not self.poisson_me or not self.poisson_int:
				if link[1] in self.node_ts_start[link[0]]:
					condition = True
				else:
					condition = False
				if link[0] in (self.node_ts_prime_start[link[1]] if self.directed else self.node_ts_start[link[1]]):
					condition_prime = True
				else:
					condition_prime = False
			## Loop over all the observed arrival times
			q = self.equal_start[link] if self.discrete else 1
			for k in range(self.nij[link]):
				self.zeta[link][k] = 0
				if self.main_effects:
					self.zeta[link][k] += alpha + beta
					## Use node_ts_edges and node_ts_prime_edges to check whether the given link corresponds to the first event on the edge
					## Add only if the event is not the first on the node
					if not self.poisson_me:
						if k >= q * condition:
							if self.hawkes_me:
								self.zeta[link][k] += mu * self.psi[link][k] 
							else:
								self.zeta[link][k] += mu * np.exp(-(mu+phi) * self.A_bar[link][k - q * condition])
						if k >= q * condition_prime:
							if self.hawkes_me:	
								self.zeta[link][k] += mu_prime * self.psi_prime[link][k]
							else:
								self.zeta[link][k] += mu_prime * np.exp(-(mu_prime+phi_prime) * self.A_bar_prime[link][k - q * condition_prime])
				if self.interactions:
					self.zeta[link][k] += np.sum(gamma * gamma_prime)
					## Do not add for the first events on the edge
					if k >= q and not self.poisson_int:
						self.zeta[link][k] += np.sum(nu * nu_prime * (self.psi_tilde[link][k] if self.hawkes_int else np.exp(-(nu+theta) * (nu_prime+theta_prime) * self.A_diff[link][k-q])))
			if verbose:
				prop += self.nij[link]
				print("\r+++ Percentage of processed links +++ {:0.2f}%".format(prop / self.m * 100), end="")
		if verbose:
			print("")

	## EM algorithm
	def em_optimise(self, max_iter=100, tolerance=1e-4):
		if not self.tau_zero or not self.full_links:
			raise ValueError('This EM algorithm can only be run when *all* links are *potentially* active. Explicit contraints on edges are in development.')
		if self.main_effects and not self.hawkes_me:
			raise ValueError('This EM algorithm can only be run for Hawkes and Poisson processes (not general Markov processes).')
		if self.interactions and not self.hawkes_int:
			raise ValueError('This EM algorithm can only be run for Hawkes and Poisson processes (not general Markov processes).')
		if not self.directed:
			raise ValueError('This EM algorithm can only be run for directed graphs (including bipartite graphs).')
		## Initialisation parameters
		if self.main_effects:
			self.alpha_tilde = np.copy(self.alpha)
			self.beta_tilde = np.copy(self.beta)
			if not self.poisson_me:
				self.mu_tilde = 1 / (1 + self.phi / self.mu)
				self.phi_tilde = self.phi + self.mu
				self.mu_prime_tilde = 1 / (1 + self.phi_prime / self.mu_prime)
				self.phi_prime_tilde = self.phi_prime + self.mu_prime
		if self.interactions:
			self.gamma_tilde = np.copy(self.gamma)
			self.gamma_prime_tilde = np.copy(self.gamma_prime)
			if not self.poisson_int:
				self.nu_tilde = 1 / (1 + self.theta / self.nu)
				self.theta_tilde = self.theta + self.nu
				self.nu_prime_tilde = 1 / (1 + self.theta_prime / self.nu_prime)
				self.theta_prime_tilde = self.theta_prime + self.nu_prime
		## Obtain the Q matrices for calculating responsibilities
		Q = {}; Q_prime = {}; Q_tilde = {}
		for link in self.A:
			if self.main_effects:
				## Consider only positive differences between the arrival times
				Q[link] = np.subtract.outer(self.A[link], self.node_ts[link[0]])
				Q[link] *= (Q[link] > 0)
				## Repeat for destination
				Q_prime[link] = np.subtract.outer(self.A[link], self.node_ts_prime[link[1]])
				Q_prime[link] *= (Q_prime[link] > 0)
			if self.interactions:
				Q_tilde[link] = np.subtract.outer(self.A[link], self.A[link])
				Q_tilde[link] *= (Q_tilde[link] > 0)
		## Define the reparametrised parameters
		if self.main_effects:
			self.csi_alpha = {}
			self.csi_beta = {} 
			if not self.poisson_me:
				self.zeta_alpha = {}
				self.zeta_beta = {}
		if self.interactions:
			self.csi_gamma = {}
			if not self.poisson_int:
				self.zeta_gamma = {}
		## Obtain the numerators for M-step for alpha and beta (identical at each iteration)
		if self.main_effects:
			den_alpha = (self.n if not self.bipartite else self.n2) * self.T
			den_beta = self.n * self.T 
		## Initialise iterations and vector for likelihood
		ll = []
		iteration = 0
		## Criterion
		tcrit = True
		## Loop until the tolerance criterion is satisfied
		while tcrit and iteration < max_iter:
			print("\r+++ Iteration {:d} +++".format(iteration+1),end="")
			iteration += 1
			## E-step: calculate responsibilities
			for link in self.A:
				## zval = np.array(list(self.zeta[link].values()))
				if self.main_effects:
					self.csi_alpha[link] = self.alpha_tilde[link[0]] ## / zval
					self.csi_beta[link] = self.beta_tilde[link[1]] ## / zval	
					if not self.poisson_me:
						self.zeta_alpha[link] = self.mu[link[0]] * np.exp(-Q[link] * self.phi_tilde[link[0]]) * (Q[link] > 0) ## / zval.reshape(-1,1)
						self.zeta_beta[link] = self.mu_prime[link[1]] * np.exp(-Q_prime[link] * self.phi_prime_tilde[link[1]]) * (Q_prime[link] > 0) ## / zval.reshape(-1,1)
				if self.interactions:
					self.csi_gamma[link] = self.gamma_tilde[link[0]] * self.gamma_prime_tilde[link[1]] ## / zval
					if not self.poisson_int:
						self.zeta_gamma[link] = self.nu[link[0]] * self.nu_prime[link[1]] * np.tile(np.ones((self.nij[link],self.nij[link])),(self.D,1,1))
						if self.D > 1: 
							for q in range(self.D):
								self.zeta_gamma[link][q] *= np.exp(-self.theta_tilde[link[0],q] * self.theta_prime_tilde[link[1],q] * Q_tilde[link]) * (Q_tilde[link] > 0) 
								## self.zeta_gamma[link][q] /= zval
						else: 
							self.zeta_gamma[link][0] *= np.exp(-self.theta_tilde[link[0]] * self.theta_prime_tilde[link[1]] * Q_tilde[link]) * (Q_tilde[link] > 0)
							self.zeta_gamma[link] = self.zeta_gamma[link][0] ## / zval
				# Renormalise if necessary
				norming_constant = 0
				if self.main_effects: 
					norming_constant += self.csi_alpha[link] + self.csi_beta[link]
					if not self.poisson_me:
						norming_constant += self.zeta_alpha[link].sum(axis=1) + self.zeta_beta[link].sum(axis=1) 
				if self.interactions:
					norming_constant += self.csi_gamma[link] 
					if not self.poisson_int:
						norming_constant = norming_constant + self.zeta_gamma[link].sum(axis=1)
				if self.main_effects:
					self.csi_alpha[link] /= norming_constant
					self.csi_beta[link] /= norming_constant
					if not self.poisson_me:
						self.zeta_alpha[link] /= norming_constant.reshape(-1,1)
						self.zeta_beta[link] /= norming_constant.reshape(-1,1)
				if self.interactions:
					if self.D > 1:
						self.csi_gamma[link] /= norming_constant.reshape(1,-1)
					else:
						self.csi_gamma[link] = self.csi_gamma[link] / norming_constant
					if not self.poisson_int:
						self.zeta_gamma[link] /= np.tile(norming_constant.reshape(-1,1),len(norming_constant))
			## Pre-allocate arrays for M-step
			if self.main_effects:
				num_alpha = np.zeros(self.n if not self.bipartite else self.n1)
				num_beta = np.zeros(self.n if not self.bipartite else self.n2)
				if not self.poisson_me:
					num_mu_phi = np.zeros(self.n if not self.bipartite else self.n1)
					den_mu = np.zeros(self.n if not self.bipartite else self.n1)
					den_phi = np.zeros(self.n if not self.bipartite else self.n1)
					num_mu_phi_prime = np.zeros(self.n if not self.bipartite else self.n2)
					den_mu_prime = np.zeros(self.n if not self.bipartite else self.n2)
					den_phi_prime = np.zeros(self.n if not self.bipartite else self.n2)
			if self.interactions:
				num_gamma = np.zeros((self.n if not self.bipartite else self.n1, self.D))
				den_gamma = np.sum(self.gamma_prime_tilde * self.T)
				if not self.poisson_int:
					if self.D > 1:
						num_nu_theta = np.zeros((self.n if not self.bipartite else self.n1, self.D))
						den_nu = np.zeros((self.n if not self.bipartite else self.n1, self.D))
						den_theta = np.zeros((self.n if not self.bipartite else self.n1, self.D))
						num_nu_theta_prime = np.zeros((self.n if not self.bipartite else self.n2, self.D))
						den_nu_prime = np.zeros((self.n if not self.bipartite else self.n2, self.D))
						den_theta_prime = np.zeros((self.n if not self.bipartite else self.n2, self.D))
					else:
						num_nu_theta = np.zeros(self.n if not self.bipartite else self.n1)
						den_nu = np.zeros(self.n if not self.bipartite else self.n1)
						den_theta = np.zeros(self.n if not self.bipartite else self.n1)
						num_nu_theta_prime = np.zeros(self.n if not self.bipartite else self.n2)
						den_nu_prime = np.zeros(self.n if not self.bipartite else self.n2)
						den_theta_prime = np.zeros(self.n if not self.bipartite else self.n2)
			## Loop over links for numerators for most parameters except theta
			for link in self.A:
				if self.main_effects:
					num_alpha[link[0]] += np.sum(self.csi_alpha[link])
					num_beta[link[1]] += np.sum(self.csi_beta[link])
					if not self.poisson_me:
						num_mu_phi[link[0]] += np.sum(self.zeta_alpha[link])
						num_mu_phi_prime[link[1]] += np.sum(self.zeta_beta[link])
				if self.interactions:
					num_gamma[link[0]] += np.sum(self.csi_gamma[link])
					if not self.poisson_int:
						if self.D > 1:
							num_nu_theta[link[0]] += np.sum(self.zeta_gamma[link],axis=(1,2))
							num_nu_theta_prime[link[1]] += np.sum(self.zeta_gamma[link],axis=(1,2))
							for q in range(self.D):
								den_nu[link[0],q] += self.nu_prime_tilde[link[1],q] * np.sum(1 - np.exp(-self.theta_tilde[link[0],q] * self.theta_prime_tilde[link[1],q] * (self.T - self.A[link])))
						else:
							num_nu_theta[link[0]] += np.sum(self.zeta_gamma[link])
							num_nu_theta_prime[link[1]] += np.sum(self.zeta_gamma[link])
							den_nu[link[0]] += self.nu_prime_tilde[link[1]] * np.sum(1 - np.exp(-self.theta_tilde[link[0]] * self.theta_prime_tilde[link[1]] * (self.T - self.A[link])))
			## M-step: maximise expected complete data log-likelihood
			if self.main_effects:
				self.alpha_tilde = num_alpha / den_alpha
				self.beta_tilde = num_beta / den_beta
			if self.interactions:
				self.gamma_tilde = num_gamma / den_gamma
				den_gamma_prime = np.sum(self.gamma_tilde * self.T)
				self.gamma_prime_tilde = num_gamma / den_gamma_prime
			## Update mu and mu_prime
			if self.main_effects and not self.poisson_me:
				for i in self.node_ts:
					den_mu[i] = (self.n if not self.bipartite else self.n2) * np.sum(1 - np.exp(-self.phi_tilde[i] * (self.T - self.node_ts[i])))
				self.mu_tilde = num_mu_phi / den_mu
				for j in self.node_ts_prime:
					den_mu_prime[j] = self.n * np.sum(1 - np.exp(-self.phi_prime_tilde[j] * (self.T - self.node_ts_prime[j])))
				self.mu_prime_tilde = num_mu_phi_prime / den_mu_prime
			## Update nu
			if self.interactions and not self.poisson_int:
				self.nu_tilde = num_nu_theta / den_nu
			## Loop on nodes for denominators for phi and phi_prime
			if self.main_effects and not self.poisson_me:
				for node in self.node_ts:
					den_phi[node] += (self.n2 if self.bipartite else self.n) * self.mu_tilde[node] * np.sum((self.T - self.node_ts[node]) * np.exp(-self.phi_tilde[node] * (self.T - self.node_ts[node])))
				for node in self.node_ts_prime:
					den_phi_prime[node] += self.n * self.mu_prime_tilde[node] * np.sum((self.T - self.node_ts_prime[node]) * np.exp(-self.phi_prime_tilde[node] * (self.T - self.node_ts_prime[node])))
			## Loop for nu_prime
			if self.interactions and not self.poisson_int:
				for link in self.A:
					if self.interactions and not self.poisson_int:
						den_nu_prime[link[1]] += self.nu_tilde[link[0]] * np.sum(1 - np.exp(-self.theta_tilde[link[0]] * self.theta_prime_tilde[link[1]] * (self.T - self.A[link])))
				self.nu_prime_tilde = num_nu_theta_prime / den_nu_prime
			## Loop on edges for denominators for phi, phi_prime, theta and theta_prime (only first part)
			if (self.main_effects and not self.poisson_me) or (self.interactions and not self.poisson_int):
				for link in self.A:
					if self.main_effects and not self.poisson_me:
						den_phi[link[0]] += np.sum(np.multiply(Q[link], self.zeta_alpha[link]))
						den_phi_prime[link[1]] += np.sum(np.multiply(Q_prime[link], self.zeta_beta[link]))
					if self.interactions and not self.poisson_int:
						if self.D > 1:
							vv = np.sum(np.multiply(Q_tilde[link], self.zeta_gamma[link]), axis=(1,2))
							vv21 = np.multiply(self.nu_tilde[link[0]], self.nu_prime_tilde[link[1]])
							vv22 = np.multiply(self.T - self.A[link], np.exp(-np.outer(np.multiply(self.theta_tilde[link[0]], self.theta_prime_tilde[link[1]]), self.T - self.A[link])))
							vv2 = np.sum(np.multiply(vv21, vv22))
						else:
							vv = np.sum(np.multiply(Q_tilde[link], self.zeta_gamma[link]))
							vv21 = np.multiply(np.multiply(self.nu_tilde[link[0]], self.nu_prime_tilde[link[1]]), self.theta_prime_tilde[link[1]])
							vv22 = np.multiply(self.T - self.A[link], np.exp(-self.theta_tilde[link[0]] * self.theta_prime_tilde[link[1]] * (self.T - self.A[link])))
							vv2 = np.sum(np.multiply(vv21, vv22))
						den_theta[link[0]] += vv + vv2
						den_theta_prime[link[1]] += vv
				## Update phi and phi_prime
				if self.main_effects and not self.poisson_me:
					self.phi_tilde = num_mu_phi / den_phi
					self.phi_prime_tilde = num_mu_phi_prime / den_phi_prime
				## Update theta
				if self.interactions and not self.poisson_int:
					self.theta_tilde = num_nu_theta / den_theta
			## Update theta_prime
			if self.interactions and not self.poisson_int:
				for link in self.A:
					vv21 = np.multiply(np.multiply(self.nu_tilde[link[0]], self.nu_prime_tilde[link[1]]), self.theta_tilde[link[0]])
					if self.D > 1:
						vv22 = np.multiply(self.T - self.A[link], np.exp(-np.outer(np.multiply(self.theta_tilde[link[0]], self.theta_prime_tilde[link[1]]), self.T - self.A[link])))
					else:
						vv22 = np.multiply(self.T - self.A[link], np.exp(-self.theta_tilde[link[0]] * self.theta_prime_tilde[link[1]] * (self.T - self.A[link])))
					vv2 = np.sum(np.multiply(vv21, vv22))
					den_theta_prime[link[1]] += vv2
				self.theta_prime_tilde = num_nu_theta / den_theta_prime
			## Convert to likelihood parametrisation
			if self.main_effects:
				self.alpha = np.copy(self.alpha_tilde)
				self.beta = np.copy(self.beta_tilde)
				if not self.poisson_me:
					self.mu = np.multiply(self.mu_tilde, self.phi_tilde)
					self.phi = self.phi_tilde - self.mu
					self.mu_prime = np.multiply(self.mu_prime_tilde, self.phi_prime_tilde)
					self.phi_prime = self.phi_prime_tilde - self.mu_prime
			if self.interactions:
				self.gamma = np.copy(self.gamma_tilde) if self.D > 1 else self.gamma_tilde[:,0]
				self.gamma_prime = np.copy(self.gamma_prime_tilde) if self.D > 1 else self.gamma_prime_tilde[:,0]
				if not self.poisson_int:
					self.nu = np.multiply(self.nu_tilde, self.theta_tilde)
					self.theta = self.theta_tilde - self.nu
					self.nu_prime = np.multiply(self.nu_prime_tilde, self.theta_prime_tilde)
					self.theta_prime = self.theta_prime_tilde - self.nu_prime
			## Setup likelihood calculations
			self.likelihood_calculation_setup(verbose=False)			
			## Calculate likelihood for	evaluating convergence
			if ((self.interactions and self.hawkes_int) or (self.main_effects and self.hawkes_me)) and (not self.poisson_me or not self.poisson_int):
				self.psi_calculation(verbose=False)
			## Calculate zeta
			self.zeta_calculation(verbose=False)
			## Calculate compensator
			self.compensator_T()
			## Use zeta to calculate the likelihood correctly
			log_likelihood = 0.0
			for link in self.A:
				log_likelihood += np.sum(np.log(list(self.zeta[link].values())))
				log_likelihood -= self.Lambda_T[link]
			## Add back missing elements
			if self.main_effects and self.full_links:
				log_likelihood -= (self.n2 if self.bipartite else self.n) * np.sum(self.alpha_compensator)
				log_likelihood -= (self.n1 if self.bipartite else self.n) * np.sum(self.beta_compensator if self.directed else self.alpha_compensator)
			if self.interactions and self.full_links:
				if self.D == 1:
					log_likelihood -= self.T * np.sum(self.gamma) * np.sum(self.gamma_prime if self.directed else self.gamma)
				else:
					log_likelihood -= self.T * np.inner(np.sum(self.gamma,axis=0), np.sum(self.gamma_prime if self.directed else self.gamma, axis=0))
			ll += [log_likelihood]
			## Calculate the criterion
			if iteration > 2 and ll[-1] - ll[-2] > 0:
				tcrit = (np.abs((ll[-1] - ll[-2]) / ll[-2]) > tolerance)
		print("")
		return ll

	## Calculation of the compensator at time T (useful for the log-likelihood) - Approximation for the discrete process
	def compensator_T(self):
		self.Lambda_T = {}
		## Main effects if full links
		if self.full_links:
			if self.main_effects:
				self.alpha_compensator = self.alpha * self.T
				if self.directed:
					self.beta_compensator = self.beta * self.T
				if not self.poisson_me:
					for i in range(self.n1 if self.bipartite else self.n):
						mu = self.mu[i]
						phi = self.phi[i]
						if i in self.node_ts:
							if self.hawkes_me:
								self.alpha_compensator[i] -= mu / (mu+phi) * np.sum(np.exp(-(mu+phi) * (self.T - self.node_ts[i])) - 1)
							else:
								self.alpha_compensator[i] -= mu / (mu+phi) * np.sum((np.exp(-(mu+phi) * np.diff(np.append(self.node_ts[i],self.T))) - 1))
					if self.directed:
						for j in range(self.n2 if self.bipartite else self.n):
							mu_prime = self.mu_prime[j]
							phi_prime = self.phi_prime[j]
							if j in self.node_ts_prime:
								if self.hawkes_me:
									self.beta_compensator[j] -= mu_prime / (mu_prime+phi_prime) * np.sum(np.exp(-(mu_prime+phi_prime) * (self.T - self.node_ts_prime[j])) - 1)
								else:
									self.beta_compensator[j] -= mu_prime / (mu_prime+phi_prime) * np.sum((np.exp(-(mu_prime + phi_prime) * \
												np.diff(np.append(self.node_ts_prime[j],self.T))) - 1))
		for link in self.A:
			self.Lambda_T[link] = 0
			## Select parameters
			if self.main_effects and not self.poisson_me:
				mu = self.mu[link[0]]
				mu_prime = self.mu_prime[link[1]] if self.directed else self.mu[link[1]]
				phi = self.phi[link[0]]
				phi_prime = self.phi_prime[link[1]] if self.directed else self.phi[link[1]]
			if self.interactions and not self.poisson_int:
				nu = self.nu[link[0]]
				nu_prime = self.nu_prime[link[1]] if self.directed else self.nu[link[1]]
				theta = self.theta[link[0]]
				theta_prime = self.theta_prime[link[1]] if self.directed else self.theta[link[1]]
			if not self.full_links:	
				## Update the main effects
				if self.main_effects:
					self.Lambda_T[link] += (self.alpha[link[0]] + (self.beta[link[1]] if self.directed else self.alpha[link[1]])) * (self.T - self.Tau[link])
					if not self.poisson_me:
						if self.hawkes_me:
							self.Lambda_T[link] -= mu / (mu+phi) * np.sum(np.exp(-(mu+phi) * (self.T - self.node_ts[link[0]])) - 
														np.exp(-(mu+phi) * np.maximum(0, self.Tau[link] - self.node_ts[link[0]])))
							self.Lambda_T[link] -= mu_prime / (mu_prime+phi_prime) * np.sum(np.exp(-(mu_prime+phi_prime) * (self.T - \
														(self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]))) - \
														np.exp(-(mu_prime+phi_prime) * np.maximum(0, self.Tau[link] - \
														(self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]))))
						else:
							zero_out = (self.node_ts[link[0]] >= self.Tau[link])
							self.Lambda_T[link] -= mu / (mu+phi) * np.sum((np.exp(-(mu+phi) * np.diff(np.append(self.node_ts[link[0]],self.T))) - 1)[zero_out])
							zero_out_prime = ((self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]) >= self.Tau[link])
							self.Lambda_T[link] -= mu_prime / (mu_prime+phi_prime) * np.sum((np.exp(-(mu_prime + phi_prime) * \
									np.diff(np.append(self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]],self.T))) - 1)[zero_out_prime])
			## Update the interactions
			if self.interactions:
				if self.D == 1:
					if not self.full_links:
						self.Lambda_T[link] += (self.gamma[link[0]] * (self.gamma_prime[link[1]] if self.directed else self.gamma[link[1]])) * (self.T - self.Tau[link])
					if not self.poisson_int:
						if self.hawkes_int:
							self.Lambda_T[link] -= (nu * nu_prime) / ((nu+theta) * (nu_prime+theta_prime)) * np.sum(np.exp(-(nu+theta) * (nu_prime+theta_prime) * (self.T - self.A[link])) - 1)
						else:
							self.Lambda_T[link] -= (nu * nu_prime) / ((nu+theta) * (nu_prime+theta_prime)) * np.sum(np.exp(-(nu+theta) * (nu_prime+theta_prime) * self.A_diff[link]) - 1)
							self.Lambda_T[link] -= (nu * nu_prime) / ((nu+theta) * (nu_prime+theta_prime)) * (np.exp(-(nu+theta) * (nu_prime+theta_prime) * (self.T - self.A[link][-1])) - 1)
				else:
					if not self.full_links:
						self.Lambda_T[link] += np.sum(self.gamma[link[0]] * (self.gamma_prime[link[1]] if self.directed else self.gamma[link[1]])) * (self.T - self.Tau[link])
					if not self.poisson_int:
						if self.hawkes_int:
							self.Lambda_T[link] -= np.sum([(nu[k] * nu_prime[k]) / ((nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k])) * np.sum(np.exp(-(nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k]) * (self.T - self.A[link])) - 1) for k in range(self.D)])
						else:
							self.Lambda_T[link] -= np.sum([(nu[k] * nu_prime[k]) / ((nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k])) * np.sum(np.exp(-(nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k]) * self.A_diff[link]) - 1) for k in range(self.D)])
							self.Lambda_T[link] -= np.sum([(nu[k] * nu_prime[k]) / ((nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k])) * (np.exp(-(nu[k]+theta[k]) * (nu_prime[k]+theta_prime[k]) * (self.T - self.A[link][-1])) - 1) for k in range(self.D)])

	## Calculation of gradients
	def gradient(self, prior_penalisation=False, verbose=True):
		if verbose:
			prop = 0
		## Initialise rows and columns to create COO sparse matrices
		rows = []
		cols = []
		if self.main_effects:
			vals = []
			if not self.poisson_me:
				vals_mu = []
				vals_phi = []
				if self.directed:
					vals_mu_prime = []
					vals_phi_prime = []
		if self.interactions:
			if self.D > 1:
				rows_int = []
				cols_int = []
				dims_int = []
			vals_gamma = []
			if not self.poisson_int:
				vals_nu = []
				vals_theta = []
			if self.directed:
				vals_gamma_prime = []
				if not self.poisson_int:
					vals_nu_prime = []
					vals_theta_prime = []
		## Calculate the components separately for full links 
		if self.full_links and self.main_effects:
			if not self.poisson_me:
				mu_component = np.zeros(self.n1 if self.bipartite else self.n)
				phi_component = np.zeros(self.n1 if self.bipartite else self.n)
				for i in range(self.n1 if self.bipartite else self.n):
					## Obtain parameters
					mu = self.mu[i]
					phi = self.phi[i]
					if i in self.node_ts:
						if self.hawkes_me:
							## Updates for mu and phi
							t_diff = self.T - self.node_ts[i]
							sum_1 = np.sum(np.exp(-(mu+phi) * t_diff) - 1)
							sum_2 = np.sum((np.multiply(t_diff, np.exp(-(mu+phi) * t_diff))))
							mu_component[i] = phi / ((mu+phi) ** 2) * sum_1 - mu / (mu+phi) * sum_2
							phi_component[i] = - mu / ((mu+phi) ** 2) * sum_1 - 1 / (mu+phi) * sum_2
						else:
							t_diff = np.diff(np.append(self.node_ts[i],self.T))
							sum_1 = np.sum((np.exp(-(mu+phi) * t_diff) - 1))
							sum_2 = np.sum(np.multiply(t_diff,np.exp(-(mu+phi) * t_diff)))
							mu_component[i] = phi / ((mu+phi) ** 2) * sum_1 - mu / (mu+phi) * sum_2
							phi_component[i] = - mu / ((mu+phi) ** 2) * sum_1 - 1 / (mu+phi) * sum_2
				mu_prime_component = np.zeros(self.n2 if self.bipartite else self.n)
				phi_prime_component = np.zeros(self.n2 if self.bipartite else self.n)
				if self.directed:
					for j in range(self.n2 if self.bipartite else self.n):	
						mu_prime = self.mu_prime[j]
						phi_prime = self.phi_prime[j]
						if j in self.node_ts_prime:
							if self.hawkes_me:
								## Repeat for mu_prime and phi_prime
								t_diff_prime = self.T - self.node_ts_prime[j]
								sum_1_prime = np.sum(np.exp(-(mu_prime+phi_prime) * t_diff_prime) - 1)
								sum_2_prime = np.sum(np.multiply(t_diff_prime, np.exp(-(mu_prime+phi_prime) * t_diff_prime)))
								mu_prime_component[j] = phi_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - mu_prime / (mu_prime+phi_prime) * sum_2_prime
								phi_prime_component[j] = - mu_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - 1 / (mu_prime+phi_prime) * sum_2_prime
							else:
								t_diff_prime = np.diff(np.append(self.node_ts_prime[j],self.T))
								sum_1_prime = np.sum((np.exp(-(mu_prime+phi_prime) * t_diff_prime) - 1))
								sum_2_prime = np.sum(np.multiply(t_diff_prime,np.exp(-(mu_prime+phi_prime) * t_diff_prime)))
								mu_prime_component[j] = phi_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - mu_prime / (mu_prime+phi_prime) * sum_2_prime
								phi_prime_component[j] = - mu_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - 1 / (mu_prime+phi_prime) * sum_2_prime
		## Calculate gradient from sparse matrices (indexed by edge)
		for link in self.A:
			## Update rows and columns
			rows += [link[0]]
			cols += [link[1]]
			if not self.directed:
				rows += [link[1]]
				cols += [link[0]]
			if self.main_effects:
				## Updates for alpha and beta
				kk = 0 if self.full_links else (self.T - self.Tau[link])
				vals += (1 if self.directed else 2) * [- kk + np.sum([1.0 / self.zeta[link][k] for k in range(self.nij[link])])]
				if not self.poisson_me:
					## Obtain parameters
					mu = self.mu[link[0]]
					mu_prime = self.mu_prime[link[1]] if self.directed else self.mu[link[1]]
					phi = self.phi[link[0]]
					phi_prime = self.phi_prime[link[1]] if self.directed else self.phi[link[1]]
					if self.hawkes_me:
						## Updates for mu and phi
						psi_sum = np.sum([self.psi[link][k] / self.zeta[link][k] for k in range(self.nij[link])])
						psi_derivative_sum = np.sum([self.psi_derivative[link][k] / self.zeta[link][k] for k in range(self.nij[link])])
						if not self.full_links:
							t_diff = self.T - self.node_ts[link[0]]
							t_diff_tau = np.maximum(0, self.Tau[link] - self.node_ts[link[0]])
							## zero_out = (self.node_ts[link[0]] >= self.Tau[link]).astype(int)
							sum_1 = np.sum(np.exp(-(mu+phi) * t_diff) - np.exp(-(mu+phi) * t_diff_tau))
							sum_2 = np.sum((np.multiply(t_diff, np.exp(-(mu+phi) * t_diff)) - np.multiply(t_diff_tau,np.exp(-(mu+phi) * t_diff_tau))))
							vals_mu += [phi / ((mu+phi) ** 2) * sum_1 - mu / (mu+phi) * sum_2 + psi_sum + mu * psi_derivative_sum]
							vals_phi += [- mu / ((mu+phi) ** 2) * sum_1 - 1 / (mu+phi) * sum_2 + mu * psi_derivative_sum]
						else:
							vals_mu += [psi_sum + mu * psi_derivative_sum]
							vals_phi += [mu * psi_derivative_sum]
						## Repeat for mu_prime and phi_prime
						psi_prime_sum = np.sum([self.psi_prime[link][k] / self.zeta[link][k] for k in range(self.nij[link])])
						psi_prime_derivative_sum = np.sum([self.psi_prime_derivative[link][k] / self.zeta[link][k] for k in range(self.nij[link])])
						if not self.full_links:
							t_diff_prime = self.T - (self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]])
							t_diff_tau_prime = np.maximum(0, self.Tau[link] - (self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]))
							## zero_out_prime = ((self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]) >= self.Tau[link]).astype(int)
							sum_1_prime = np.sum(np.exp(-(mu_prime+phi_prime) * t_diff_prime) - np.exp(-(mu_prime+phi_prime) * t_diff_tau_prime))
							sum_2_prime = np.sum((np.multiply(t_diff_prime, np.exp(-(mu_prime+phi_prime) * t_diff_prime)) - \
												np.multiply(t_diff_tau_prime, np.exp(-(mu_prime+phi_prime) * t_diff_tau_prime))))
							res_mu = [phi_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - mu_prime / (mu_prime+phi_prime) * sum_2_prime + psi_prime_sum + mu_prime * psi_prime_derivative_sum]
							res_phi = [- mu_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - 1 / (mu_prime+phi_prime) * sum_2_prime + mu_prime * psi_prime_derivative_sum]
						else:
							res_mu = [psi_prime_sum + mu_prime * psi_prime_derivative_sum]
							res_phi = [mu_prime * psi_prime_derivative_sum]
					else:
						## Updates for mu and phi
						condition = (self.nij[link] != len(self.A_bar[link])) * (self.equal_start[link] if self.discrete else 1)
						t_bar = self.A_bar[link]
						t_diff = np.diff(np.append(self.node_ts[link[0]],self.T))
						psi_sum = np.sum([np.exp(-(mu+phi) * t_bar[k - condition]) / self.zeta[link][k] for k in range(condition, self.nij[link])])
						psi_derivative_sum = np.sum([np.exp(-(mu+phi) * t_bar[k - condition]) * t_bar[k - condition] / self.zeta[link][k] for k in range(condition, self.nij[link])])
						if not self.full_links:
							zero_out = (self.node_ts[link[0]] >= self.Tau[link])
							sum_1 = np.sum((np.exp(-(mu+phi) * t_diff) - 1)[zero_out])
							sum_2 = np.sum(np.multiply(t_diff,np.exp(-(mu+phi) * t_diff))[zero_out])
							vals_mu += [phi / ((mu+phi) ** 2) * sum_1 - mu / (mu+phi) * sum_2 + psi_sum - mu * psi_derivative_sum]
							vals_phi += [- mu / ((mu+phi) ** 2) * sum_1 - 1 / (mu+phi) * sum_2 - mu * psi_derivative_sum]
						else:
							vals_mu += [psi_sum - mu * psi_derivative_sum]
							vals_phi += [- mu * psi_derivative_sum]
						## Repeat for mu_prime and phi_prime 
						condition_prime = (self.nij[link] != len(self.A_bar_prime[link])) * (self.equal_start[link] if self.discrete else 1)
						t_bar_prime = self.A_bar_prime[link]
						t_diff_prime = np.diff(np.append(self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]],self.T))
						psi_prime_sum = np.sum([np.exp(-(mu_prime+phi_prime) * t_bar_prime[k-condition_prime]) / self.zeta[link][k] for k in range(condition_prime, self.nij[link])])
						psi_prime_derivative_sum = np.sum([np.exp(-(mu_prime+phi_prime) * t_bar_prime[k-condition_prime]) * t_bar_prime[k-condition_prime] / self.zeta[link][k] for k in range(condition_prime, self.nij[link])])
						if not self.full_links:
							zero_out_prime = ((self.node_ts_prime[link[1]] if self.directed else self.node_ts[link[1]]) >= self.Tau[link])
							sum_1_prime = np.sum((np.exp(-(mu_prime+phi_prime) * t_diff_prime) - 1)[zero_out_prime])
							sum_2_prime = np.sum(np.multiply(t_diff_prime,np.exp(-(mu_prime+phi_prime) * t_diff_prime))[zero_out_prime])
							res_mu = [phi_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - mu_prime / (mu_prime+phi_prime) * sum_2_prime + psi_prime_sum - mu_prime * psi_prime_derivative_sum]
							res_phi = [- mu_prime / ((mu_prime+phi_prime) ** 2) * sum_1_prime - 1 / (mu_prime+phi_prime) * sum_2_prime - mu_prime * psi_prime_derivative_sum]
						else:
							res_mu = [psi_prime_sum - mu_prime * psi_prime_derivative_sum]
							res_phi = [- mu_prime * psi_prime_derivative_sum]
					## If the graph is directed, add to vals_parameter_prime, otherwise (undirected graph) to vals_parameter
					if self.directed:
						vals_mu_prime += res_mu
						vals_phi_prime += res_phi
					else:
						vals_mu += res_mu
						vals_phi += res_phi		
			if self.interactions:
				## Update rows and columns
				if self.D > 1:
					rows_int += [link[0]]*self.D
					cols_int += [link[1]]*self.D
					dims_int += list(range(self.D))
					if not self.directed:
						rows_int += [link[1]]*self.D
						cols_int += [link[0]]*self.D
						dims_int += list(range(self.D))
				## Updates for gamma
				gamma = self.gamma[link[0]]
				gamma_prime = self.gamma_prime[link[1]] if self.directed else self.gamma[link[1]]
				if self.D == 1:
					vals_gamma += [gamma_prime * (- ((self.T - self.Tau[link]) if not self.full_links else 0) + \
										np.sum([1.0 / self.zeta[link][k] for k in range(self.nij[link])]))]
					res_gamma = [gamma * (- ((self.T - self.Tau[link]) if not self.full_links else 0) + \
										np.sum([1.0 / self.zeta[link][k] for k in range(self.nij[link])]))]
				else:
					vals_gamma += list(gamma_prime * (- ((self.T - self.Tau[link]) if not self.full_links else 0) + \
										np.sum([1.0 / self.zeta[link][k] for k in range(self.nij[link])])))
					res_gamma = list(gamma * (- ((self.T - self.Tau[link]) if not self.full_links else 0) + \
										np.sum([1.0 / self.zeta[link][k] for k in range(self.nij[link])])))
				if self.directed:
					vals_gamma_prime += res_gamma
				else:
					vals_gamma += res_gamma
				if not self.poisson_int:
					## Obtain the parameters
					nu = self.nu[link[0]]
					nu_prime = self.nu_prime[link[1]] if self.directed else self.nu[link[1]]
					theta = self.theta[link[0]]
					theta_prime = self.theta_prime[link[1]] if self.directed else self.theta[link[1]]
					q = self.equal_start[link] if self.discrete else 1
					if self.hawkes_int:
						## Psi summations
						psi_sum = np.sum([self.psi_tilde[link][k] / self.zeta[link][k] for k in range(q,self.nij[link])],axis=0)
						psi_derivative_sum = np.sum([self.psi_tilde_derivative[link][k] / self.zeta[link][k] for k in range(q,self.nij[link])],axis=0)
						psi_prime_derivative_sum = np.sum([self.psi_tilde_derivative_prime[link][k] / self.zeta[link][k] for k in range(q,self.nij[link])],axis=0)
						## Updates for nu and theta
						t_diff = self.T - np.array(self.A[link])
						if self.D == 1:
							sum_1 = np.sum(np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff) - 1)
							sum_2 = np.sum(np.multiply(t_diff,np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff)))
							vals_nu += [nu_prime / (nu+theta) * (theta / (nu+theta) / (nu_prime+theta_prime) * sum_1 - nu * sum_2) + nu_prime * (psi_sum + nu * psi_derivative_sum)]
							vals_theta += [- nu * nu_prime / (nu+theta) * (sum_1 / (nu + theta) / (nu_prime + theta_prime) + sum_2) + nu * psi_derivative_sum]
							## Repeat for nu_prime and theta_prime
							res_nu = [nu / (nu_prime+theta_prime) * (theta_prime / (nu_prime+theta_prime) / (nu+theta) * sum_1 - nu_prime * sum_2) + nu * (psi_sum + nu_prime * psi_prime_derivative_sum)]
							res_theta = [- nu_prime * nu / (nu_prime+theta_prime) * (sum_1 / (nu_prime + theta_prime) / (nu + theta) + sum_2) + nu_prime * psi_prime_derivative_sum]
						else:
							sum_1 = np.sum([np.exp(-(nu+theta) * (nu_prime+theta_prime) * tt) - 1 for tt in t_diff], axis=0)
							sum_2 = np.sum([tt * np.exp(-(nu+theta) * (nu_prime+theta_prime) * tt) for tt in t_diff], axis=0)
							vals_nu += list(nu_prime / (nu+theta) * (theta / (nu+theta) / (nu_prime+theta_prime) * sum_1 - nu * sum_2) + nu_prime * (psi_sum + nu * psi_derivative_sum))
							vals_theta += list(- nu * nu_prime / (nu+theta) * (sum_1 / (nu + theta) / (nu_prime + theta_prime) + sum_2) + nu * psi_derivative_sum)
							## Repeat for nu_prime and theta_prime
							res_nu = list(nu / (nu_prime+theta_prime) * (theta_prime / (nu_prime+theta_prime) / (nu+theta) * sum_1 - nu_prime * sum_2) + nu * (psi_sum + nu_prime * psi_prime_derivative_sum))
							res_theta = list(- nu_prime * nu / (nu_prime+theta_prime) * (sum_1 / (nu_prime + theta_prime) / (nu + theta) + sum_2) + nu_prime * psi_prime_derivative_sum)
					else:
						## Updates for nu and theta
						t_diff = np.append(self.A_diff[link], self.T - self.A[link][-1])
						## Calculate the updates
						if self.D == 1:
							sum_1 = np.sum(np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff) - 1)
							sum_2 = np.sum(np.multiply(t_diff, np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff)))
							psi_sum = np.sum([np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff[k-q]) / self.zeta[link][k] for k in range(q, self.nij[link])]) 
							psi_derivative_sum = np.sum([t_diff[k-q] * np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff[k-q]) / self.zeta[link][k] for k in range(q, self.nij[link])])
							## Calculate the updates
							vals_nu += [nu_prime / (nu+theta) * (theta / (nu+theta) / (nu_prime+theta_prime) * sum_1 - nu * sum_2) + nu_prime * (psi_sum - nu * (nu_prime+theta_prime) * psi_derivative_sum)]
							vals_theta += [- nu * nu_prime / (nu+theta) * (sum_1 / (nu + theta) / (nu_prime + theta_prime) + sum_2) - nu * nu_prime * (nu_prime + theta_prime) * psi_derivative_sum]
							## Repeat for nu_prime and theta_prime
							res_nu = [nu / (nu_prime+theta_prime) * (theta_prime / (nu_prime+theta_prime) / (nu+theta) * sum_1 - nu_prime * sum_2) + nu * (psi_sum - nu_prime * (nu+theta) * psi_derivative_sum)]
							res_theta = [- nu_prime * nu / (nu_prime+theta_prime) * (sum_1 / (nu_prime + theta_prime) / (nu + theta) + sum_2) - nu * nu_prime * (nu + theta) * psi_derivative_sum]
						else:
							sum_1 = np.sum([np.exp(-(nu+theta) * (nu_prime+theta_prime) * tt) - 1 for tt in t_diff], axis=0)
							sum_2 = np.sum([tt * np.exp(-(nu+theta) * (nu_prime+theta_prime) * tt) for tt in t_diff],axis=0)
							psi_sum = np.sum([np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff[k-q]) / self.zeta[link][k] for k in range(q, self.nij[link])], axis=0) 
							psi_derivative_sum = np.sum([t_diff[k-q] * np.exp(-(nu+theta) * (nu_prime+theta_prime) * t_diff[k-q]) / self.zeta[link][k] for k in range(q, self.nij[link])], axis=0)
							vals_nu += list(nu_prime / (nu+theta) * (theta / (nu+theta) / (nu_prime+theta_prime) * sum_1 - nu * sum_2) + nu_prime * (psi_sum - nu * (nu_prime+theta_prime) * psi_derivative_sum))
							vals_theta += list(- nu * nu_prime / (nu+theta) * (sum_1 / (nu + theta) / (nu_prime + theta_prime) + sum_2) - nu * nu_prime * (nu_prime + theta_prime) * psi_derivative_sum)
							## Repeat for nu_prime and theta_prime
							res_nu = list(nu / (nu_prime+theta_prime) * (theta_prime / (nu_prime+theta_prime) / (nu+theta) * sum_1 - nu_prime * sum_2) + nu * (psi_sum - nu_prime * (nu+theta) * psi_derivative_sum))
							res_theta = list(- nu_prime * nu / (nu_prime+theta_prime) * (sum_1 / (nu_prime + theta_prime) / (nu + theta) + sum_2) - nu * nu_prime * (nu + theta) * psi_derivative_sum)		
					## If the graph is directed, add to vals_parameter_prime, otherwise (undirected graph) to vals_parameter
					if self.directed:
						vals_nu_prime += res_nu
						vals_theta_prime += res_theta
					else:
						vals_nu += res_nu
						vals_theta += res_theta
			if verbose:
				prop += self.nij[link]
				print("\r+++ Percentage of processed links +++ {:0.2f}%".format(prop / self.m * 100), end="")
		if verbose:
			print("")
		## Calculate the gradients for the main effects
		if self.main_effects:
			## For undirected graphs, sum along the columns (by construction of the matrices)
			Z = coo_matrix((vals, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
			## Baseline parameters
			self.grad_alpha = np.array(Z.sum(axis=1)).flatten()
			if self.full_links:
				self.grad_alpha -= self.T * (self.n2 if self.bipartite else self.n)
			if self.directed:
				self.grad_beta = np.array(Z.sum(axis=0)).flatten()
				if self.full_links:
					self.grad_beta -= self.T * (self.n1 if self.bipartite else self.n)
			if not self.poisson_me:
				## Excitation function parameters
				Z = coo_matrix((vals_mu, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
				self.grad_mu = np.array(Z.sum(axis=1)).flatten()
				if self.full_links:
					self.grad_mu += mu_component * (self.n2 if self.bipartite else self.n)
				Z = coo_matrix((vals_phi, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
				self.grad_phi = np.array(Z.sum(axis=1)).flatten()
				if self.full_links:
					self.grad_phi += phi_component * (self.n2 if self.bipartite else self.n)
				## Parameters of the directed graph
				if self.directed:
					Z = coo_matrix((vals_mu_prime, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
					self.grad_mu_prime = np.array(Z.sum(axis=0)).flatten()
					if self.full_links:
						self.grad_mu_prime += mu_prime_component * (self.n1 if self.bipartite else self.n)
					Z = coo_matrix((vals_phi_prime, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
					self.grad_phi_prime = np.array(Z.sum(axis=0)).flatten()
					if self.full_links:
						self.grad_phi_prime += phi_prime_component * (self.n1 if self.bipartite else self.n)
			if verbose:
				print("+++ Updating the parameters for the main effects +++")
		## Calculate the gradients for the interactions
		if self.interactions:
			if self.D == 1:
				## Baseline parameters
				Z = coo_matrix((vals_gamma, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
				self.grad_gamma = np.array(Z.sum(axis=1)).flatten()
				if self.full_links:
					self.grad_gamma -= self.T * np.sum(self.gamma_prime if self.directed else self.gamma)
				if not self.poisson_int:
					## Excitation function
					Z = coo_matrix((vals_nu, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
					self.grad_nu = np.array(Z.sum(axis=1)).flatten()
					Z = coo_matrix((vals_theta, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
					self.grad_theta = np.array(Z.sum(axis=1)).flatten()
				## Parameters of the directed graph
				if self.directed:
					Z = coo_matrix((vals_gamma_prime, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
					self.grad_gamma_prime = np.array(Z.sum(axis=0)).flatten()
					if self.full_links:
						self.grad_gamma_prime -= self.T * np.sum(self.gamma)
					if not self.poisson_int:
						Z = coo_matrix((vals_nu_prime, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
						self.grad_nu_prime = np.array(Z.sum(axis=0)).flatten()
						Z = coo_matrix((vals_theta_prime, (rows, cols)), shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n))
						self.grad_theta_prime = np.array(Z.sum(axis=0)).flatten()
			else:
				## Baseline parameters
				Z = COO([rows_int, cols_int, dims_int], vals_gamma, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
				self.grad_gamma = Z.sum(axis=1).todense()
				if self.full_links:
					self.grad_gamma -= self.T * np.sum(self.gamma_prime if self.directed else self.gamma, axis=0)
				if not self.poisson_int:
					## Excitation function
					Z = COO((rows_int, cols_int, dims_int), vals_nu, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
					self.grad_nu = Z.sum(axis=1).todense()
					Z = COO((rows_int, cols_int, dims_int), vals_theta, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
					self.grad_theta = Z.sum(axis=1).todense()
				## Parameters of the directed graph
				if self.directed:
					Z = COO((rows_int, cols_int, dims_int), vals_gamma_prime, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
					self.grad_gamma_prime = Z.sum(axis=0).todense()
					if self.full_links:
						self.grad_gamma_prime -= self.T * np.sum(self.gamma, axis=0)
					if not self.poisson_int:
						Z = COO((rows_int, cols_int, dims_int), vals_nu_prime, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
						self.grad_nu_prime = Z.sum(axis=0).todense()
						Z = COO((rows_int, cols_int, dims_int), vals_theta_prime, shape=(self.n1 if self.bipartite else self.n, self.n2 if self.bipartite else self.n, self.D))
						self.grad_theta_prime = Z.sum(axis=0).todense()
			if verbose:
				prop += self.nij[link]
				print("+++ Updating the parameters for the interactions +++")
		## Add penalty terms from the prior
		if prior_penalisation:
			if self.main_effects:
				self.grad_alpha = scipy.stats.gamma.logpdf(self.alpha, a=self.tau[0], scale=1/self.tau[1]) + self.grad_alpha
				if not self.poisson_me:	
					self.grad_mu = scipy.stats.gamma.logpdf(self.mu, a=self.csi[0], scale=1/self.csi[1]) + self.grad_mu
					self.grad_phi = scipy.stats.gamma.logpdf(self.phi, a=self.kappa[0], scale=1/self.kappa[1]) + self.grad_phi
				if self.directed:
					self.grad_beta = scipy.stats.gamma.logpdf(self.beta, a=self.tau_prime[0], scale=1/self.tau_prime[1]) + self.grad_beta
					if not self.poisson_me:	
						self.grad_mu_prime = scipy.stats.gamma.logpdf(self.mu_prime, a=self.csi_prime[0], scale=1/self.csi_prime[1]) + self.grad_mu_prime
						self.grad_phi_prime = scipy.stats.gamma.logpdf(self.phi_prime, a=self.kappa_prime[0], scale=1/self.kappa_prime[1]) + self.grad_phi_prime
			if self.interactions:
				self.grad_gamma = scipy.stats.gamma.logpdf(self.gamma, a=self.tau_tilde[0], scale=1/self.tau_tilde[1]) + self.grad_gamma
				if not self.poisson_int:
					self.grad_nu = scipy.stats.gamma.logpdf(self.nu, a=self.csi_tilde[0], scale=1/self.csi_tilde[1]) + self.grad_nu
					self.grad_theta = scipy.stats.gamma.logpdf(self.theta, a=self.kappa_tilde[0], scale=1/self.kappa_tilde[1]) + self.grad_theta
				if self.directed:
					self.grad_gamma_prime = scipy.stats.gamma.logpdf(self.gamma_prime, a=self.tau_tilde_prime[0], scale=1/self.tau_tilde_prime[1]) + self.grad_gamma_prime
					if not self.poisson_int:
						self.grad_nu_prime = scipy.stats.gamma.logpdf(self.nu_prime, a=self.csi_tilde_prime[0], scale=1/self.csi_tilde_prime[1]) + self.grad_nu_prime
						self.grad_theta_prime = scipy.stats.gamma.logpdf(self.theta_prime, a=self.kappa_tilde_prime[0], scale=1/self.kappa_tilde_prime[1]) + self.grad_theta_prime
			## Update the prior gradients
			if self.main_effects:
				self.grad_tau = scipy.stats.gamma.logpdf(self.tau, a=self.hyper_alpha[0], scale=1/self.hyper_alpha[1])
				if not self.poisson_me:
					self.grad_csi = scipy.stats.gamma.logpdf(self.csi, a=self.hyper_mu[0], scale=1/self.hyper_mu[1])
					self.grad_kappa = scipy.stats.gamma.logpdf(self.kappa, a=self.hyper_phi[0], scale=1/self.hyper_phi[1])
				if self.directed:
					self.grad_tau_prime = scipy.stats.gamma.logpdf(self.tau_prime, a=self.hyper_beta[0], scale=1/self.hyper_beta[1])
					if not self.poisson_me:
						self.grad_csi_prime = scipy.stats.gamma.logpdf(self.csi_prime, a=self.hyper_mu_prime[0], scale=1/self.hyper_mu_prime[1])
						self.grad_kappa_prime = scipy.stats.gamma.logpdf(self.kappa_prime, a=self.hyper_phi_prime[0], scale=1/self.hyper_phi_prime[1])			
			if self.interactions:
				self.grad_tau_tilde = scipy.stats.gamma.logpdf(self.tau_tilde, a=self.hyper_gamma[0], scale=1/self.hyper_gamma[1])
				if not self.poisson_int:
					self.grad_csi_tilde = scipy.stats.gamma.logpdf(self.csi_tilde, a=self.hyper_nu[0], scale=1/self.hyper_nu[1])
					self.grad_kappa_tilde = scipy.stats.gamma.logpdf(self.kappa_tilde, a=self.hyper_theta[0], scale=1/self.hyper_theta[1])
				if self.directed:
					self.grad_tau_tilde_prime = scipy.stats.gamma.logpdf(self.tau_tilde_prime, a=self.hyper_gamma_prime[0], scale=1/self.hyper_gamma_prime[1])
					if not self.poisson_int:
						self.grad_csi_tilde_prime = scipy.stats.gamma.logpdf(self.csi_tilde_prime, a=self.hyper_nu_prime[0], scale=1/self.hyper_nu_prime[1])
						self.grad_kappa_tilde_prime = scipy.stats.gamma.logpdf(self.kappa_tilde_prime, a=self.hyper_theta_prime[0], scale=1/self.hyper_theta_prime[1])		

	## Main function for sequential optimisation of the MEG model
	def optimise_meg(self, learning_rate=0.01, method='standard', rho=0.9, rho2=0.99, max_iter=100, tolerance=1e-4, 
						ada_epsilon=1e-8, prior_penalisation=False, verbose=True):
		## The learning rate corresponds to the parameter rho in AdaDelta (AdaDelta does not require the specification of a learning rate)
		if not isinstance(learning_rate, numbers.Number) or learning_rate <= 0:
			return ValueError("The learning rate should be a positive real number.")
		if method == 'standard':
			adaptive_rate = False
		else:
			if method in ['adagrad','adadelta','rmsprop','adam']:
				adaptive_rate = True
			else:
				return ValueError("The method should be 'standard', 'bb' or 'ada' depending on the required gradient ascent technique.")
    	## Initialise G for ADA
		if 'ada' in method or 'rms' in method:
			if self.main_effects:
				G_alpha = np.zeros(self.n1 if self.bipartite else self.n)
				if not self.poisson_me:
					G_mu = np.zeros(self.n1 if self.bipartite else self.n)
					G_phi = np.zeros(self.n1 if self.bipartite else self.n)
				if self.directed:
					G_beta = np.zeros(self.n2 if self.bipartite else self.n)
					if not self.poisson_me:
						G_mu_prime = np.zeros(self.n2 if self.bipartite else self.n)
						G_phi_prime = np.zeros(self.n2 if self.bipartite else self.n)
			if self.interactions:
				G_gamma = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
				if not self.poisson_int:
					G_nu = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
					G_theta = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
				if self.directed:
					G_gamma_prime = np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
					if not self.poisson_int:	
						G_nu_prime = np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
						G_theta_prime =  np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
			## Define the corresponding quantities for the prior hyperparameters
			if prior_penalisation:
				if self.main_effects:
					G_tau = np.zeros(2)
					if not self.poisson_me:
						G_csi = np.zeros(2)
						G_kappa = np.zeros(2) 
					if self.directed:
						G_tau_prime = np.zeros(2) 
						if not self.poisson_me:
							G_csi_prime = np.zeros(2) 
							G_kappa_prime = np.zeros(2) 
				if self.interactions:
					G_tau_tilde = np.zeros(2) 
					if not self.poisson_int:
						G_csi_tilde = np.zeros(2) 
						G_kappa_tilde = np.zeros(2)
					if self.directed:
						G_tau_tilde_prime = np.zeros(2) 
						if not self.poisson_int:
							G_csi_tilde_prime = np.zeros(2) 
							G_kappa_tilde_prime = np.zeros(2) 
			## Define Delta for AdaDelta or Adam
			if method == 'adadelta' or method == 'adam':
				if rho > 1 or rho < 0:
					return ValueError('rho must be in [0,1].')
				if method == 'adam':
					if rho2 > 1 or rho2 < 0:
						return ValueError('rho2 must be in [0,1].')
				if self.main_effects:
					Delta_alpha = np.zeros(self.n1 if self.bipartite else self.n) 
					if not self.poisson_me:
						Delta_mu = np.zeros(self.n1 if self.bipartite else self.n)
						Delta_phi = np.zeros(self.n1 if self.bipartite else self.n)
					if self.directed:
						Delta_beta = np.zeros(self.n2 if self.bipartite else self.n)
						if not self.poisson_me:
							Delta_mu_prime = np.zeros(self.n2 if self.bipartite else self.n)
							Delta_phi_prime = np.zeros(self.n2 if self.bipartite else self.n)
				if self.interactions:
					Delta_gamma = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
					if not self.poisson_int:
						Delta_nu = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
						Delta_theta = np.zeros((self.n1 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n1 if self.bipartite else self.n)
					if self.directed:
						Delta_gamma_prime = np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
						if not self.poisson_int:
							Delta_nu_prime = np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
							Delta_theta_prime = np.zeros((self.n2 if self.bipartite else self.n, self.D)) if self.D > 1 else np.zeros(self.n2 if self.bipartite else self.n)
				if prior_penalisation:
					if self.main_effects:
						Delta_tau = np.zeros(2)
						if not self.poisson_me:
							Delta_csi = np.zeros(2)
							Delta_kappa = np.zeros(2) 
						if self.directed:
							Delta_tau_prime = np.zeros(2) 
							if not self.poisson_me:
								Delta_csi_prime = np.zeros(2) 
								Delta_kappa_prime = np.zeros(2) 
					if self.interactions:
						Delta_tau_tilde = np.zeros(2) 
						if not self.poisson_int:
							Delta_csi_tilde = np.zeros(2) 
							Delta_kappa_tilde = np.zeros(2)
						if self.directed:
							Delta_tau_tilde_prime = np.zeros(2) 
							if not self.poisson_int:
								Delta_csi_tilde_prime = np.zeros(2) 
								Delta_kappa_tilde_prime = np.zeros(2) 
		## Initalise vector containing the log-likelihoods
		ll = []
		## Iterate the optimisation
		print("+++ Iteration 1 +++",end="")
		## Setup likelihood calculations
		self.likelihood_calculation_setup(verbose=verbose)
		## Calculate psi
		if ((self.main_effects and self.hawkes_me) or (self.interactions and self.hawkes_int)) and (not self.poisson_me or not self.poisson_int):
			self.psi_calculation(verbose=verbose,calculate_derivative=True)
		## Calculate zeta
		self.zeta_calculation(verbose=verbose)
		## Initialise iterations
		iteration = 0
		## Criterion
		tcrit = True
		## Loop until the tolerance criterion is satisfied
		while tcrit and iteration < max_iter:
			print("\r+++ Iteration {:d} +++".format(iteration+1),end="")
			iteration += 1
			## Update the gradient
			self.gradient(prior_penalisation=prior_penalisation,verbose=verbose)
			## Calculate using gradient ascent
			if adaptive_rate:
				## Calculate AdaGrad learning rates
				if method == 'adagrad':
					if self.main_effects:
						G_alpha += np.multiply(self.grad_alpha, self.alpha) ** 2
						if not self.poisson_me:
							G_mu += np.multiply(self.grad_mu, self.mu) ** 2
							G_phi += np.multiply(self.grad_phi, self.phi) ** 2
						eta_alpha = learning_rate / np.sqrt(G_alpha + ada_epsilon)
						if not self.poisson_me:
							eta_mu = learning_rate / np.sqrt(G_mu + ada_epsilon)
							eta_phi = learning_rate / np.sqrt(G_phi + ada_epsilon)
						if self.directed:
							G_beta += np.multiply(self.grad_beta, self.beta) ** 2
							if not self.poisson_me:
								G_mu_prime += np.multiply(self.grad_mu_prime, self.mu_prime) ** 2
								G_phi_prime += np.multiply(self.grad_phi_prime, self.phi_prime) ** 2
							eta_beta = np.divide(learning_rate, np.sqrt(G_beta + ada_epsilon))
							if not self.poisson_me:
								eta_mu_prime = np.divide(learning_rate, np.sqrt(G_mu_prime + ada_epsilon))
								eta_phi_prime = np.divide(learning_rate, np.sqrt(G_phi_prime + ada_epsilon))
					if self.interactions:
						G_gamma += np.multiply(self.grad_gamma, self.gamma) ** 2
						if not self.poisson_int:
							G_nu += np.multiply(self.grad_nu, self.nu) ** 2
							G_theta += np.multiply(self.grad_theta, self.theta) ** 2
						eta_gamma = np.divide(learning_rate, np.sqrt(G_gamma + ada_epsilon))
						if not self.poisson_int:
							eta_nu = np.divide(learning_rate, np.sqrt(G_nu + ada_epsilon))
							eta_theta = np.divide(learning_rate, np.sqrt(G_theta + ada_epsilon))
						if self.directed:
							G_gamma_prime += np.multiply(self.grad_gamma_prime, self.gamma_prime) ** 2
							if not self.poisson_int:
								G_nu_prime += np.multiply(self.grad_nu_prime, self.nu_prime) ** 2
								G_theta_prime += np.multiply(self.grad_theta_prime, self.theta_prime) ** 2
							eta_gamma_prime = np.divide(learning_rate, np.sqrt(G_gamma_prime + ada_epsilon))
							if not self.poisson_int:
								eta_nu_prime = np.divide(learning_rate, np.sqrt(G_nu_prime + ada_epsilon))
								eta_theta_prime = np.divide(learning_rate, np.sqrt(G_theta_prime + ada_epsilon))
					if prior_penalisation:
						if self.main_effects:
							G_tau += np.multiply(self.grad_tau, self.tau) ** 2
							if not self.poisson_me:
								G_csi += np.multiply(self.grad_csi, self.csi) ** 2
								G_kappa += np.multiply(self.grad_kappa, self.kappa) ** 2
							eta_tau = learning_rate / np.sqrt(G_tau + ada_epsilon)
							if not self.poisson_me:
								eta_csi = learning_rate / np.sqrt(G_csi + ada_epsilon)
								eta_kappa = learning_rate / np.sqrt(G_kappa + ada_epsilon)
							if self.directed:
								G_tau_prime += np.multiply(self.grad_tau_prime, self.tau_prime) ** 2
								if not self.poisson_me:
									G_csi_prime += np.multiply(self.grad_csi_prime, self.csi_prime) ** 2
									G_kappa_prime += np.multiply(self.grad_kappa_prime, self.kappa_prime) ** 2
								eta_tau_prime = learning_rate / np.sqrt(G_tau_prime + ada_epsilon)
								if not self.poisson_me:
									eta_csi_prime = learning_rate / np.sqrt(G_csi_prime + ada_epsilon)
									eta_kappa_prime = learning_rate / np.sqrt(G_kappa_prime + ada_epsilon)
						if self.interactions:
							G_tau_tilde += np.multiply(self.grad_tau_tilde, self.tau_tilde) ** 2
							if not self.poisson_int:
								G_csi_tilde += np.multiply(self.grad_csi_tilde, self.csi_tilde) ** 2
								G_kappa_tilde += np.multiply(self.grad_kappa_tilde, self.kappa_tilde) ** 2
							eta_tau_tilde = learning_rate / np.sqrt(G_tau_tilde + ada_epsilon)
							if not self.poisson_int:
								eta_csi_tilde = learning_rate / np.sqrt(G_csi_tilde + ada_epsilon)
								eta_kappa_tilde = learning_rate / np.sqrt(G_kappa_tilde + ada_epsilon)
							if self.directed:
								G_tau_tilde_prime += np.multiply(self.grad_tau_tilde_prime, self.tau_tilde_prime) ** 2
								if not self.poisson_int:
									G_csi_tilde_prime += np.multiply(self.grad_csi_tilde_prime, self.csi_tilde_prime) ** 2
									G_kappa_tilde_prime += np.multiply(self.grad_kappa_tilde_prime, self.kappa_tilde_prime) ** 2
								eta_tau_tilde_prime = learning_rate / np.sqrt(G_tau_tilde_prime + ada_epsilon)
								if not self.poisson_int:
									eta_csi_tilde_prime = learning_rate / np.sqrt(G_csi_tilde_prime + ada_epsilon)
									eta_kappa_tilde_prime = learning_rate / np.sqrt(G_kappa_tilde_prime + ada_epsilon)
				## Calculate AdaDelta (and RMSprop) learning rates
				if method == 'adadelta' or method == 'rmsprop':
					if self.main_effects:
						G_alpha = rho * G_alpha + (1-rho) * np.multiply(self.grad_alpha, self.alpha) ** 2
						if not self.poisson_me:
							G_mu = rho * G_mu + (1-rho) * np.multiply(self.grad_mu, self.mu) ** 2
							G_phi = rho * G_phi + (1-rho) * np.multiply(self.grad_phi, self.phi) ** 2
						if method == 'adadelta':
							Delta_alpha = (rho * Delta_alpha + (1-rho) * (np.log(self.alpha) - np.log(alpha_old)) ** 2) if iteration > 2 else 1.0
							if not self.poisson_me:
								Delta_mu = (rho * Delta_mu + (1-rho) * (np.log(self.mu) - np.log(mu_old)) ** 2) if iteration > 2 else 1.0
								Delta_phi = (rho * Delta_phi + (1-rho) * (np.log(self.phi) - np.log(phi_old)) ** 2) if iteration > 2 else 1.0
							eta_alpha = np.divide(np.sqrt(Delta_alpha + ada_epsilon), np.sqrt(G_alpha + ada_epsilon))
							if not self.poisson_me:
								eta_mu = np.divide(np.sqrt(Delta_mu + ada_epsilon), np.sqrt(G_mu + ada_epsilon))
								eta_phi = np.divide(np.sqrt(Delta_phi + ada_epsilon), np.sqrt(G_phi + ada_epsilon))
						else:
							eta_alpha = np.divide(learning_rate, np.sqrt(G_alpha + ada_epsilon))
							if not self.poisson_me:
								eta_mu = np.divide(learning_rate, np.sqrt(G_mu + ada_epsilon))
								eta_phi = np.divide(learning_rate, np.sqrt(G_phi + ada_epsilon))
						if self.directed:
							G_beta = rho * G_beta + (1-rho) * np.multiply(self.grad_beta, self.beta) ** 2
							if not self.poisson_me:
								G_mu_prime = rho * G_mu_prime + (1-rho) * np.multiply(self.grad_mu_prime, self.mu_prime) ** 2
								G_phi_prime = rho * G_phi_prime + (1-rho) * np.multiply(self.grad_phi_prime, self.phi_prime) ** 2
							if method == 'adadelta':
								Delta_beta = (rho * Delta_beta + (1-rho) * (np.log(self.beta) - np.log(beta_old)) ** 2) if iteration > 2 else 1.0
								if not self.poisson_me:
									Delta_mu_prime = (rho * Delta_mu_prime + (1-rho) * (np.log(self.mu_prime) - np.log(mu_prime_old)) ** 2) if iteration > 2 else 1.0
									Delta_phi_prime = (rho * Delta_phi_prime + (1-rho) * (np.log(self.phi_prime) - np.log(phi_prime_old)) ** 2) if iteration > 2 else 1.0
								eta_beta = np.divide(np.sqrt(Delta_beta + ada_epsilon), np.sqrt(G_beta + ada_epsilon))
								if not self.poisson_me:
									eta_mu_prime = np.divide(np.sqrt(Delta_mu_prime + ada_epsilon), np.sqrt(G_mu_prime + ada_epsilon))
									eta_phi_prime = np.divide(np.sqrt(Delta_phi_prime + ada_epsilon), np.sqrt(G_phi_prime + ada_epsilon)) 
							else:
								eta_beta = np.divide(learning_rate, np.sqrt(G_beta + ada_epsilon))
								if not self.poisson_me:
									eta_mu_prime = np.divide(learning_rate, np.sqrt(G_mu_prime + ada_epsilon))
									eta_phi_prime = np.divide(learning_rate, np.sqrt(G_phi_prime + ada_epsilon)) 
					if self.interactions:
						G_gamma = rho * G_gamma + (1-rho) * np.multiply(self.grad_gamma, self.gamma) ** 2
						if not self.poisson_int:
							G_nu = rho * G_nu + (1-rho) * np.multiply(self.grad_nu, self.nu) ** 2
							G_theta = rho * G_theta + (1-rho) * np.multiply(self.grad_theta, self.theta) ** 2
						if method == 'adadelta':
							Delta_gamma = (rho * Delta_beta + (1-rho) * (np.log(self.gamma) - np.log(gamma_old)) ** 2) if iteration > 2 else 1.0
							if not self.poisson_int:
								Delta_nu = (rho * Delta_mu_prime + (1-rho) * (np.log(self.nu) - np.log(nu_old)) ** 2) if iteration > 2 else 1.0
								Delta_theta = (rho * Delta_phi_prime + (1-rho) * (np.log(self.theta) - np.log(theta_old)) ** 2) if iteration > 2 else 1.0
							eta_gamma = np.divide(np.sqrt(Delta_gamma + ada_epsilon), np.sqrt(G_gamma + ada_epsilon))
							if not self.poisson_int:
								eta_nu = np.divide(np.sqrt(Delta_nu + ada_epsilon), np.sqrt(G_nu + ada_epsilon))
								eta_theta = np.divide(np.sqrt(Delta_theta + ada_epsilon), np.sqrt(G_theta + ada_epsilon))
						else:
							eta_gamma = np.divide(learning_rate, np.sqrt(G_gamma + ada_epsilon))
							if not self.poisson_int:
								eta_nu = np.divide(learning_rate, np.sqrt(G_nu + ada_epsilon))
								eta_theta = np.divide(learning_rate, np.sqrt(G_theta + ada_epsilon))
						if self.directed:
							G_gamma_prime = rho * G_gamma_prime + (1-rho) * np.multiply(self.grad_gamma_prime, self.gamma_prime) ** 2
							if not self.poisson_int:
								G_nu_prime = rho * G_nu_prime + (1-rho) * np.multiply(self.grad_nu_prime, self.nu_prime) ** 2
								G_theta_prime = rho * G_theta_prime + (1-rho) * np.multiply(self.grad_theta_prime, self.theta_prime) ** 2
							if method == 'adadelta':
								Delta_gamma_prime = (rho * Delta_gamma_prime + (1-rho) * (np.log(self.gamma_prime) - np.log(gamma_prime_old)) ** 2) if iteration > 2 else 1.0
								if not self.poisson_int:
									Delta_nu_prime = (rho * Delta_nu_prime + (1-rho) * (np.log(self.nu_prime) - np.log(nu_prime_old)) ** 2) if iteration > 2 else 1.0
									Delta_theta_prime = (rho * Delta_theta_prime + (1-rho) * (np.log(self.theta_prime) - np.log(theta_prime_old)) ** 2) if iteration > 2 else 1.0							
								eta_gamma_prime = np.divide(np.sqrt(Delta_gamma_prime + ada_epsilon), np.sqrt(G_gamma_prime + ada_epsilon))
								if not self.poisson_int:
									eta_nu_prime = np.divide(np.sqrt(Delta_nu_prime + ada_epsilon), np.sqrt(G_nu_prime + ada_epsilon))
									eta_theta_prime = np.divide(np.sqrt(Delta_theta_prime + ada_epsilon), np.sqrt(G_theta_prime + ada_epsilon))
							else:
								eta_gamma_prime = np.divide(learning_rate, np.sqrt(G_gamma_prime + ada_epsilon))
								if not self.poisson_int:
									eta_nu_prime = np.divide(learning_rate, np.sqrt(G_nu_prime + ada_epsilon))
									eta_theta_prime = np.divide(learning_rate, np.sqrt(G_theta_prime + ada_epsilon))
					## Prior penalisation
					if prior_penalisation:
						if self.main_effects:
							G_tau = rho * G_tau + (1-rho) * np.multiply(self.grad_tau, self.tau) ** 2
							if not self.poisson_me:
								G_csi = rho * G_csi + (1-rho) * np.multiply(self.grad_csi, self.csi) ** 2
								G_kappa = rho * G_kappa + (1-rho) * np.multiply(self.grad_kappa, self.kappa) ** 2
							if method == 'adadelta':
								Delta_tau = rho * Delta_tau + (1-rho) * (np.log(self.tau) - np.log(tau_old)) ** 2
								if not self.poisson_me:
									Delta_csi = rho * Delta_csi + (1-rho) * (np.log(self.csi) - np.log(csi_old)) ** 2
									Delta_kappa = rho * Delta_kappa + (1-rho) * (np.log(self.kappa) - np.log(kappa_old)) ** 2
								eta_tau = np.divide(np.sqrt(Delta_tau + ada_epsilon), np.sqrt(G_tau + ada_epsilon))
								if not self.poisson_me:
									eta_csi = np.divide(np.sqrt(Delta_csi + ada_epsilon), np.sqrt(G_csi + ada_epsilon))
									eta_kappa = np.divide(np.sqrt(Delta_kappa + ada_epsilon), np.sqrt(G_kappa + ada_epsilon))
							else:
								eta_tau = np.divide(learning_rate, np.sqrt(G_tau + ada_epsilon))
								if not self.poisson_me:
									eta_csi = np.divide(learning_rate, np.sqrt(G_csi + ada_epsilon))
									eta_kappa = np.divide(learning_rate, np.sqrt(G_kappa + ada_epsilon))
							if self.directed:
								G_tau_prime = rho * G_tau_prime + (1-rho) * np.multiply(self.grad_tau_prime, self.tau_prime) ** 2
								if not self.poisson_me:
									G_csi_prime = rho * G_csi_prime + (1-rho) * np.multiply(self.grad_csi_prime, self.csi_prime) ** 2
									G_kappa_prime = rho * G_kappa_prime + (1-rho) * np.multiply(self.grad_kappa_prime, self.kappa_prime) ** 2
								if method == 'adadelta':
									Delta_tau_prime = rho * Delta_tau_prime + (1-rho) * (np.log(self.tau_prime) - np.log(tau_prime_old)) ** 2
									if not self.poisson_me:
										Delta_csi_prime = rho * Delta_csi_prime + (1-rho) * (np.log(self.csi_prime) - np.log(csi_prime_old)) ** 2
										Delta_kappa_prime = rho * Delta_kappa_prime + (1-rho) * (np.log(self.kappa_prime) - np.log(kappa_prime_old)) ** 2
									eta_tau_prime = np.divide(np.sqrt(Delta_tau_prime + ada_epsilon), np.sqrt(G_tau_prime + ada_epsilon))
									if not self.poisson_me:
										eta_csi_prime = np.divide(np.sqrt(Delta_csi_prime + ada_epsilon), np.sqrt(G_csi_prime + ada_epsilon))
										eta_kappa_prime = np.divide(np.sqrt(Delta_kappa_prime + ada_epsilon), np.sqrt(G_kappa_prime + ada_epsilon))
								else:
									eta_tau_prime = np.divide(learning_rate, np.sqrt(G_tau_prime + ada_epsilon))
									if not self.poisson_me:
										eta_csi_prime = np.divide(learning_rate, np.sqrt(G_csi_prime + ada_epsilon))
										eta_kappa_prime = np.divide(learning_rate, np.sqrt(G_kappa_prime + ada_epsilon))
						if self.interactions:
							G_tau_tilde = rho * G_tau_tilde + (1-rho) * np.multiply(self.grad_tau_tilde, self.tau_tilde) ** 2
							if not self.poisson_int:
								G_csi_tilde = rho * G_csi_tilde + (1-rho) * np.multiply(self.grad_csi_tilde, self.csi_tilde) ** 2
								G_kappa_tilde = rho * G_kappa_tilde + (1-rho) * np.multiply(self.grad_kappa_tilde, self.kappa_tilde) ** 2
							if method == 'adadelta':
								Delta_tau_tilde = rho * Delta_tau_tilde + (1-rho) * (np.log(self.tau_tilde) - np.log(tau_tilde_old)) ** 2
								if not self.poisson_int:
									Delta_csi_tilde = rho * Delta_csi_tilde + (1-rho) * (np.log(self.csi_tilde) - np.log(csi_tilde_old)) ** 2
									Delta_kappa_tilde = rho * Delta_kappa_tilde + (1-rho) * (np.log(self.kappa_tilde) - np.log(kappa_tilde_old)) ** 2
								eta_tau_tilde = np.divide(np.sqrt(Delta_tau_tilde + ada_epsilon), np.sqrt(G_tau_tilde + ada_epsilon))
								if not self.poisson_int:
									eta_csi_tilde = np.divide(np.sqrt(Delta_csi_tilde + ada_epsilon), np.sqrt(G_csi_tilde + ada_epsilon))
									eta_kappa_tilde = np.divide(np.sqrt(Delta_kappa_tilde + ada_epsilon), np.sqrt(G_kappa_tilde + ada_epsilon))
							else:
								eta_tau_tilde = np.divide(learning_rate, np.sqrt(G_tau_tilde + ada_epsilon))
								if not self.poisson_int:
									eta_csi_tilde = np.divide(learning_rate, np.sqrt(G_csi_tilde + ada_epsilon))
									eta_kappa_tilde = np.divide(learning_rate, np.sqrt(G_kappa_tilde + ada_epsilon))
							if self.directed:
								G_tau_tilde_prime = rho * G_tau_tilde_prime + (1-rho) * np.multiply(self.grad_tau_tilde_prime, self.tau_tilde_prime) ** 2
								if not self.poisson_int:
									G_csi_tilde_prime = rho * G_csi_tilde_prime + (1-rho) * np.multiply(self.grad_csi_tilde_prime, self.csi_tilde_prime) ** 2
									G_kappa_tilde_prime = rho * G_kappa_tilde_prime + (1-rho) * np.multiply(self.grad_kappa_tilde_prime, self.kappa_tilde_prime) ** 2
								if method == 'adadelta':
									Delta_tau_tilde_prime = rho * Delta_tau_tilde_prime + (1-rho) * (np.log(self.tau_tilde_prime) - np.log(tau_tilde_prime_old)) ** 2
									if not self.poisson_int:
										Delta_csi_tilde_prime = rho * Delta_csi_tilde_prime + (1-rho) * (np.log(self.csi_tilde_prime) - np.log(csi_tilde_prime_old)) ** 2
										Delta_kappa_tilde_prime = rho * Delta_kappa_tilde_prime + (1-rho) * (np.log(self.kappa_tilde_prime) - np.log(kappa_tilde_prime_old)) ** 2
									eta_tau_tilde_prime = np.divide(np.sqrt(Delta_tau_tilde_prime + ada_epsilon), np.sqrt(G_tau_tilde_prime + ada_epsilon))
									if not self.poisson_int:
										eta_csi_tilde_prime = np.divide(np.sqrt(Delta_csi_tilde_prime + ada_epsilon), np.sqrt(G_csi_tilde_prime + ada_epsilon))
										eta_kappa_tilde_prime = np.divide(np.sqrt(Delta_kappa_tilde_prime + ada_epsilon), np.sqrt(G_kappa_tilde_prime + ada_epsilon))
								else:
									eta_tau_tilde_prime = np.divide(learning_rate, np.sqrt(G_tau_tilde_prime + ada_epsilon))
									if not self.poisson_int:
										eta_csi_tilde_prime = np.divide(learning_rate, np.sqrt(G_csi_tilde_prime + ada_epsilon))
										eta_kappa_tilde_prime = np.divide(learning_rate, np.sqrt(G_kappa_tilde_prime + ada_epsilon))
				## Calculate Adam learning rates
				if method == 'adam':
					if self.main_effects:
						G_alpha = rho * G_alpha + (1-rho) * np.multiply(self.grad_alpha, self.alpha)
						if not self.poisson_me:
							G_mu = rho * G_mu + (1-rho) * np.multiply(self.grad_mu, self.mu)
							G_phi = rho * G_phi + (1-rho) * np.multiply(self.grad_phi, self.phi)
						Delta_alpha = rho2 * Delta_alpha + (1-rho2) * np.multiply(self.grad_alpha, self.alpha) ** 2
						if not self.poisson_me:
							Delta_mu = rho2 * Delta_mu + (1-rho2) * np.multiply(self.grad_mu, self.mu) ** 2
							Delta_phi = rho2 * Delta_phi + (1-rho2) * np.multiply(self.grad_phi, self.phi) ** 2
						eta_alpha = np.divide(learning_rate * G_alpha / (1 - rho ** iteration), np.sqrt(Delta_alpha / (1 - rho2 ** iteration)) + ada_epsilon)
						if not self.poisson_me:
							eta_mu = np.divide(learning_rate * G_mu / (1 - rho ** iteration), np.sqrt(Delta_mu / (1 - rho2 ** iteration)) + ada_epsilon)
							eta_phi = np.divide(learning_rate * G_phi / (1 - rho ** iteration), np.sqrt(Delta_phi / (1 - rho2 ** iteration)) + ada_epsilon)
						if self.directed:
							G_beta = rho * G_beta + (1-rho) * np.multiply(self.grad_beta, self.beta)
							if not self.poisson_me:
								G_mu_prime = rho * G_mu_prime + (1-rho) * np.multiply(self.grad_mu_prime, self.mu_prime)
								G_phi_prime = rho * G_phi_prime + (1-rho) * np.multiply(self.grad_phi_prime, self.phi_prime)
							Delta_beta = rho2 * Delta_beta + (1-rho2) * np.multiply(self.grad_beta, self.beta) ** 2
							if not self.poisson_me:
								Delta_mu_prime = rho2 * Delta_mu_prime + (1-rho2) * np.multiply(self.grad_mu_prime, self.mu_prime) ** 2
								Delta_phi_prime = rho2 * Delta_phi_prime + (1-rho2) * np.multiply(self.grad_phi_prime, self.phi_prime) ** 2
							eta_beta = np.divide(learning_rate * G_beta / (1 - rho ** iteration), np.sqrt(Delta_beta / (1 - rho2 ** iteration)) + ada_epsilon)
							if not self.poisson_me:
								eta_mu_prime = np.divide(learning_rate * G_mu_prime / (1 - rho ** iteration), np.sqrt(Delta_mu_prime / (1 - rho2 ** iteration)) + ada_epsilon)
								eta_phi_prime = np.divide(learning_rate * G_phi_prime / (1 - rho ** iteration), np.sqrt(Delta_phi_prime / (1 - rho2 ** iteration)) + ada_epsilon)
					if self.interactions:
						G_gamma = rho * G_gamma + (1-rho) * np.multiply(self.grad_gamma, self.gamma)
						if not self.poisson_int:
							G_nu = rho * G_nu + (1-rho) * np.multiply(self.grad_nu, self.nu)
							G_theta = rho * G_theta + (1-rho) * np.multiply(self.grad_theta, self.theta)
						Delta_gamma = rho2 * Delta_gamma + (1-rho2) * np.multiply(self.grad_gamma, self.gamma) ** 2
						if not self.poisson_int:
							Delta_nu = rho2 * Delta_nu + (1-rho2) * np.multiply(self.grad_nu, self.nu) ** 2
							Delta_theta = rho2 * Delta_theta + (1-rho2) * np.multiply(self.grad_theta, self.theta) ** 2
						eta_gamma = np.divide(learning_rate * G_gamma / (1 - rho ** iteration), np.sqrt(Delta_gamma / (1 - rho2 ** iteration)) + ada_epsilon)
						if not self.poisson_int:
							eta_nu = np.divide(learning_rate * G_nu / (1 - rho ** iteration), np.sqrt(Delta_nu / (1 - rho2 ** iteration)) + ada_epsilon)
							eta_theta = np.divide(learning_rate * G_theta / (1 - rho ** iteration), np.sqrt(Delta_theta / (1 - rho2 ** iteration)) + ada_epsilon)
						if self.directed:
							G_gamma_prime = rho * G_gamma_prime + (1-rho) * np.multiply(self.grad_gamma_prime, self.gamma_prime)
							if not self.poisson_int:
								G_nu_prime = rho * G_nu_prime + (1-rho) * np.multiply(self.grad_nu_prime, self.nu_prime)
								G_theta_prime = rho * G_theta_prime + (1-rho) * np.multiply(self.grad_theta_prime, self.theta_prime)
							Delta_gamma_prime = rho2 * Delta_gamma_prime + (1-rho2) * np.multiply(self.grad_gamma_prime, self.gamma_prime) ** 2
							if not self.poisson_int:
								Delta_nu_prime = rho2 * Delta_nu_prime + (1-rho2) * np.multiply(self.grad_nu_prime, self.nu_prime) ** 2
								Delta_theta_prime = rho2 * Delta_theta_prime + (1-rho2) * np.multiply(self.grad_theta_prime, self.theta_prime) ** 2						
							eta_gamma_prime = np.divide(learning_rate * G_gamma_prime / (1 - rho ** iteration), np.sqrt(Delta_gamma_prime / (1 - rho2 ** iteration)) + ada_epsilon)
							if not self.poisson_int:
								eta_nu_prime = np.divide(learning_rate * G_nu_prime / (1 - rho ** iteration), np.sqrt(Delta_nu_prime / (1 - rho2 ** iteration)) + ada_epsilon)
								eta_theta_prime = np.divide(learning_rate * G_theta_prime / (1 - rho ** iteration), np.sqrt(Delta_theta_prime / (1 - rho2 ** iteration)) + ada_epsilon)
					if prior_penalisation:
						if self.main_effects:
							G_tau = rho * G_tau + (1-rho) * np.multiply(self.grad_tau, self.tau)
							if not self.poisson_me:
								G_csi = rho * G_csi + (1-rho) * np.multiply(self.grad_csi, self.csi)
								G_kappa = rho * G_kappa + (1-rho) * np.multiply(self.grad_kappa, self.kappa)
							Delta_tau = rho2 * Delta_tau + (1-rho2) * np.multiply(self.grad_tau, self.tau) ** 2
							if not self.poisson_me:
								Delta_csi = rho2 * Delta_csi + (1-rho2) * np.multiply(self.grad_csi, self.csi) ** 2
								Delta_kappa = rho2 * Delta_kappa + (1-rho2) * np.multiply(self.grad_kappa, self.kappa) ** 2
							eta_tau = np.divide(learning_rate * G_tau / (1 - rho ** iteration), np.sqrt(Delta_tau / (1 - rho2 ** iteration)) + ada_epsilon)
							if not self.poisson_me:
								eta_csi = np.divide(learning_rate * G_csi / (1 - rho ** iteration), np.sqrt(Delta_csi / (1 - rho2 ** iteration)) + ada_epsilon)
								eta_kappa = np.divide(learning_rate * G_kappa / (1 - rho ** iteration), np.sqrt(Delta_kappa / (1 - rho2 ** iteration)) + ada_epsilon)
							if self.directed:
								G_tau_prime = rho * G_tau_prime + (1-rho) * np.multiply(self.grad_tau_prime, self.tau_prime)
								if not self.poisson_me:
									G_csi_prime = rho * G_csi_prime + (1-rho) * np.multiply(self.grad_csi_prime, self.csi_prime)
									G_kappa_prime = rho * G_kappa_prime + (1-rho) * np.multiply(self.grad_kappa_prime, self.kappa_prime)
								Delta_tau_prime = rho2 * Delta_tau_prime + (1-rho2) * np.multiply(self.grad_tau_prime, self.tau_prime) ** 2
								if not self.poisson_me:
									Delta_csi_prime = rho2 * Delta_csi_prime + (1-rho2) * np.multiply(self.grad_csi_prime, self.csi_prime) ** 2
									Delta_kappa_prime = rho2 * Delta_kappa_prime + (1-rho2) * np.multiply(self.grad_kappa_prime, self.kappa_prime) ** 2
								eta_tau_prime = np.divide(learning_rate * G_tau_prime / (1 - rho ** iteration), np.sqrt(Delta_tau_prime / (1 - rho2 ** iteration)) + ada_epsilon)
								if not self.poisson_me:
									eta_csi_prime = np.divide(learning_rate * G_csi_prime / (1 - rho ** iteration), np.sqrt(Delta_csi_prime / (1 - rho2 ** iteration)) + ada_epsilon)
									eta_kappa_prime = np.divide(learning_rate * G_kappa_prime / (1 - rho ** iteration), np.sqrt(Delta_kappa_prime / (1 - rho2 ** iteration)) + ada_epsilon)
						if self.interactions:
							G_tau_tilde = rho * G_tau_tilde + (1-rho) * np.multiply(self.grad_tau_tilde, self.tau_tilde)
							if not self.poisson_int:
								G_csi_tilde = rho * G_csi_tilde + (1-rho) * np.multiply(self.grad_csi_tilde, self.csi_tilde)
								G_kappa_tilde = rho * G_kappa_tilde + (1-rho) * np.multiply(self.grad_kappa_tilde, self.kappa_tilde)
							Delta_tau_tilde = rho2 * Delta_tau_tilde + (1-rho2) * np.multiply(self.grad_tau_tilde, self.tau_tilde) ** 2
							if not self.poisson_int:
								Delta_csi_tilde = rho2 * Delta_csi_tilde + (1-rho2) * np.multiply(self.grad_csi_tilde, self.csi_tilde) ** 2
								Delta_kappa_tilde = rho2 * Delta_kappa_tilde + (1-rho2) * np.multiply(self.grad_kappa_tilde, self.kappa_tilde) ** 2
							eta_tau_tilde = np.divide(learning_rate * G_tau_tilde / (1 - rho ** iteration), np.sqrt(Delta_tau_tilde / (1 - rho2 ** iteration)) + ada_epsilon)
							if not self.poisson_int:
								eta_csi_tilde = np.divide(learning_rate * G_csi_tilde / (1 - rho ** iteration), np.sqrt(Delta_csi_tilde / (1 - rho2 ** iteration)) + ada_epsilon)
								eta_kappa_tilde = np.divide(learning_rate * G_kappa_tilde / (1 - rho ** iteration), np.sqrt(Delta_kappa_tilde / (1 - rho2 ** iteration)) + ada_epsilon)
							if self.directed:
								G_tau_tilde_prime = rho * G_tau_tilde_prime + (1-rho) * np.multiply(self.grad_tau_tilde_prime, self.tau_tilde_prime)
								if not self.poisson_int:
									G_csi_tilde_prime = rho * G_csi_tilde_prime + (1-rho) * np.multiply(self.grad_csi_tilde_prime, self.csi_tilde_prime)
									G_kappa_tilde_prime = rho * G_kappa_tilde_prime + (1-rho) * np.multiply(self.grad_kappa_tilde_prime, self.kappa_tilde_prime)
								Delta_tau_tilde_prime = rho2 * Delta_tau_tilde_prime + (1-rho2) * np.multiply(self.grad_tau_tilde_prime, self.tau_tilde_prime) ** 2
								if not self.poisson_int:
									Delta_csi_tilde_prime = rho2 * Delta_csi_tilde_prime + (1-rho2) * np.multiply(self.grad_csi_tilde_prime, self.csi_tilde_prime) ** 2
									Delta_kappa_tilde_prime = rho2 * Delta_kappa_tilde_prime + (1-rho2) * np.multiply(self.grad_kappa_tilde_prime, self.kappa_tilde_prime) ** 2
								eta_tau_tilde_prime = np.divide(learning_rate * G_tau_tilde_prime / (1 - rho ** iteration), np.sqrt(Delta_tau_tilde_prime / (1 - rho2 ** iteration)) + ada_epsilon)
								if not self.poisson_int:
									eta_csi_tilde_prime = np.divide(learning_rate * G_csi_tilde_prime / (1 - rho ** iteration), np.sqrt(Delta_csi_tilde_prime / (1 - rho2 ** iteration)) + ada_epsilon)
									eta_kappa_tilde_prime = np.divide(learning_rate * G_kappa_tilde_prime / (1 - rho ** iteration), np.sqrt(Delta_kappa_tilde_prime / (1 - rho2 ** iteration)) + ada_epsilon)
			else:
				eta = learning_rate
			## Store the old values of the parameters before the update (only for BB criterion)
			if method == 'bb' or method == 'adadelta' or method == 'rmsprop' or method == 'adam':
				if self.main_effects:
					alpha_old = self.alpha
					if not self.poisson_me:
						mu_old = self.mu
						phi_old = self.phi
					if self.directed:
						beta_old = self.beta
						if not self.poisson_me:
							mu_prime_old = self.mu_prime
							phi_prime_old = self.phi_prime
				if self.interactions:
					gamma_old = self.gamma
					if not self.poisson_int:
						nu_old = self.nu
						theta_old = self.theta
					if self.directed:
						gamma_prime_old = self.gamma_prime
						if not self.poisson_int:
							nu_prime_old = self.nu_prime
							theta_prime_old = self.theta_prime
				if prior_penalisation:
					if self.main_effects:
						tau_old = self.tau
						if not self.poisson_me:
							csi_old = self.csi
							kappa_old = self.kappa
						if self.directed:
							tau_prime_old = self.tau_prime
							if not self.poisson_me:
								csi_prime_old = self.csi_prime
								kappa_prime_old = self.kappa_prime
					if self.interactions:
						tau_tilde_old = self.tau_tilde
						if not self.poisson_int:
							csi_tilde_old = self.csi_tilde
							kappa_tilde_old = self.kappa_tilde
						if self.directed:
							tau_tilde_prime_old = self.tau_tilde_prime
							if not self.poisson_int:
								csi_tilde_prime_old = self.csi_tilde_prime
								kappa_tilde_prime_old = self.kappa_tilde_prime
			## Update the parameters using gradient ascent
			if self.main_effects:
				self.alpha = np.multiply(self.alpha, np.exp(np.multiply(eta_alpha if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_alpha, self.alpha)) if method != 'adam' else eta_alpha))
				if not self.poisson_me:
					self.mu = np.multiply(self.mu, np.exp(np.multiply(eta_mu if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_mu, self.mu)) if method != 'adam' else eta_mu))
					self.phi = np.multiply(self.phi, np.exp(np.multiply(eta_phi if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_phi, self.phi)) if method != 'adam' else eta_phi))
				if self.directed:
					self.beta = np.multiply(self.beta, np.exp(np.multiply(eta_beta if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_beta, self.beta)) if method != 'adam' else eta_beta))
					if not self.poisson_me:
						self.mu_prime = np.multiply(self.mu_prime, np.exp(np.multiply(eta_mu_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_mu_prime, self.mu_prime)) if method != 'adam' else eta_mu_prime))
						self.phi_prime = np.multiply(self.phi_prime, np.exp(np.multiply(eta_phi_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_phi_prime, self.phi_prime)) if method != 'adam' else eta_phi_prime))
			if self.interactions:
				self.gamma = np.multiply(self.gamma, np.exp(np.multiply(eta_gamma if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_gamma, self.gamma)) if method != 'adam' else eta_gamma))
				if not self.poisson_int:
					self.nu = np.multiply(self.nu, np.exp(np.multiply(eta_nu if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_nu, self.nu)) if method != 'adam' else eta_nu))
					self.theta = np.multiply(self.theta, np.exp(np.multiply(eta_theta if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_theta, self.theta)) if method != 'adam' else eta_theta))
				if self.directed:
					self.gamma_prime = np.multiply(self.gamma_prime, np.exp(np.multiply(eta_gamma_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_gamma_prime, self.gamma_prime)) if method != 'adam' else eta_gamma_prime))
					if not self.poisson_int:
						self.nu_prime = np.multiply(self.nu_prime, np.exp(np.multiply(eta_nu_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_nu_prime, self.nu_prime)) if method != 'adam' else eta_nu_prime))
						self.theta_prime = np.multiply(self.theta_prime, np.exp(np.multiply(eta_theta_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_theta_prime, self.theta_prime)) if method != 'adam' else eta_theta_prime)) 
			if prior_penalisation:
				if self.main_effects:
					self.tau = np.multiply(self.tau, np.exp(np.multiply(eta_tau if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_tau, self.tau)) if method != 'adam' else eta_tau))
					if not self.poisson_me:
						self.csi = np.multiply(self.csi, np.exp(np.multiply(eta_csi if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_csi, self.csi)) if method != 'adam' else eta_csi))
						self.kappa = np.multiply(self.kappa, np.exp(np.multiply(eta_kappa if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_kappa, self.kappa)) if method != 'adam' else eta_kappa))
					if self.directed:
						self.tau_prime = np.multiply(self.tau_prime, np.exp(np.multiply(eta_tau_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_tau_prime, self.tau_prime)) if method != 'adam' else eta_tau_prime))
						if not self.poisson_me:
							self.csi_prime = np.multiply(self.csi_prime, np.exp(np.multiply(eta_csi_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_csi_prime, self.csi_prime)) if method != 'adam' else eta_csi_prime))
							self.kappa_prime = np.multiply(self.kappa_prime, np.exp(np.multiply(eta_kappa_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_kappa_prime, self.kappa_prime)) if method != 'adam' else eta_kappa_prime))
				if self.interactions:
					self.tau_tilde = np.multiply(self.tau_tilde, np.exp(np.multiply(eta_tau_tilde if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_tau_tilde, self.tau_tilde)) if method != 'adam' else eta_tau_tilde))
					if not self.poisson_int:
						self.csi_tilde = np.multiply(self.csi_tilde, np.exp(np.multiply(eta_csi_tilde if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_csi_tilde, self.csi_tilde)) if method != 'adam' else eta_csi_tilde))
						self.kappa_tilde = np.multiply(self.kappa_tilde, np.exp(np.multiply(eta_kappa_tilde if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_kappa_tilde, self.kappa_tilde)) if method != 'adam' else eta_kappa_tilde))
					if self.directed:
						self.tau_tilde_prime = np.multiply(self.tau_tilde_prime, np.exp(np.multiply(eta_tau_tilde_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_tau_tilde_prime, self.tau_tilde_prime)) if method != 'adam' else eta_tau_tilde_prime))
						if not self.poisson_int:
							self.csi_tilde_prime = np.multiply(self.csi_tilde_prime, np.exp(np.multiply(eta_csi_tilde_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_csi_tilde_prime, self.csi_tilde_prime)) if method != 'adam' else eta_csi_tilde_prime))
							self.kappa_tilde_prime = np.multiply(self.kappa_tilde_prime, np.exp(np.multiply(eta_kappa_tilde_prime if 'ada' in method or 'rms' in method else eta, np.multiply(self.grad_kappa_tilde_prime, self.kappa_tilde_prime)) if method != 'adam' else eta_kappa_tilde_prime))
			## Calculate psi
			if ((self.interactions and self.hawkes_int) or (self.main_effects and self.hawkes_me)) and (not self.poisson_me or not self.poisson_int):
				self.psi_calculation(verbose=verbose)
			## Calculate zeta
			self.zeta_calculation(verbose=verbose)
			## Calculate compensator
			self.compensator_T()
			## Use zeta to calculate the likelihood correctly
			log_likelihood = 0.0
			for link in self.A:
				log_likelihood += np.sum(np.log(list(self.zeta[link].values())))
				log_likelihood -= self.Lambda_T[link]
			## Add back missing elements
			if self.main_effects and self.full_links:
				log_likelihood -= (self.n2 if self.bipartite else self.n) * np.sum(self.alpha_compensator)
				log_likelihood -= (self.n1 if self.bipartite else self.n) * np.sum(self.beta_compensator if self.directed else self.alpha_compensator)
			if self.interactions and self.full_links:
				if self.D == 1:
					log_likelihood -= self.T * np.sum(self.gamma) * np.sum(self.gamma_prime if self.directed else self.gamma)
				else:
					log_likelihood -= self.T * np.inner(np.sum(self.gamma,axis=0), np.sum(self.gamma_prime if self.directed else self.gamma, axis=0))
			ll += [log_likelihood]
			## Calculate the criterion
			if iteration > 2 and ll[-1] - ll[-2] > 0:
				tcrit = (np.abs((ll[-1] - ll[-2]) / ll[-2]) > tolerance)
		print("")
		return ll

	## Calculation of the compensator for any t (useful for the p-values) - Approximation for the discrete process
	def compensator(self):
		self.Lambda = {}
		for link in self.A:
			self.Lambda[link] = np.zeros(self.nij[link])
			## Select parameters
			if self.main_effects and not self.poisson_me:
				mu = self.mu[link[0]]
				mu_prime = self.mu_prime[link[1]] if self.directed else self.mu[link[1]]
				phi = self.phi[link[0]]
				phi_prime = self.phi_prime[link[1]] if self.directed else self.phi[link[1]]
			if self.interactions and not self.poisson_int:
				nu = self.nu[link[0]]
				nu_prime = self.nu_prime[link[1]] if self.directed else self.nu[link[1]]
				theta = self.theta[link[0]]
				theta_prime = self.theta_prime[link[1]] if self.directed else self.theta[link[1]]
			## Update the main effects
			if self.main_effects:
				self.Lambda[link] += (self.alpha[link[0]] + (self.beta[link[1]] if self.directed else self.alpha[link[1]])) * self.A[link]
				if not self.poisson_me:
					## Initialise ni and nj_prime
					ni = 0
					nj_prime = 0
					## Calculate (again) ell_k and ell_k_prime if not Hawkes
					if not self.hawkes_me:
						## Obtain ell_k indices (connections on the given edge on the node time series)
						ell_k = np.where(self.node_ts_edges[link[0]] == link[1])[0]
						ts = self.node_ts[link[0]]
						if self.discrete:
							for k in range(len(ell_k)):
								arrival_time = ts[ell_k[k]]
								k_mod = ell_k[k]
								while ts[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k[k] = k_mod
						## Calculate the required quantities
						phi_cusum = np.zeros(self.nij[link])
						phi_cusum[0] = np.sum(np.exp(-(mu+phi) * np.diff(self.node_ts[link[0]][:(ell_k[0]+1)])) - 1)
						for k in range(1,self.nij[link]):
							phi_cusum[k] = phi_cusum[k-1] + np.sum(np.exp(-(mu+phi) * np.diff(self.node_ts[link[0]][ell_k[k-1]:(ell_k[k]+1)])) - 1)
						## Repeat for ell_k_prime
						ell_k_prime = np.where((self.node_ts_prime_edges[link[1]] if self.directed else self.node_ts_edges[link[1]]) == link[0])[0]
						ts_prime = (self.node_ts_prime if self.directed else self.node_ts)[link[1]]
						if self.discrete:
							for k in range(len(ell_k_prime)):
								arrival_time = ts_prime[ell_k_prime[k]]
								k_mod = ell_k_prime[k]
								while ts_prime[k_mod] == arrival_time:
									k_mod -= 1
									if k_mod < 0:
										break
								ell_k_prime[k] = k_mod
						## Calculate the required quantities
						phi_cusum_prime = np.zeros(self.nij[link])
						phi_cusum_prime[0] = np.sum(np.exp(-(mu_prime+phi_prime) * np.diff(self.node_ts_prime[link[1]][:(ell_k_prime[0]+1)])) - 1)
						for k in range(1,self.nij[link]):
							phi_cusum_prime[k] = phi_cusum_prime[k-1] + np.sum(np.exp(-(mu_prime+phi_prime) * np.diff(self.node_ts_prime[link[1]][ell_k_prime[k-1]:(ell_k_prime[k]+1)])) - 1)
					## Loop over the edge observations
					for k in range(self.nij[link]):
						if self.hawkes_me:
							ni += len(self.psi_times[link][k]) + 1
							nj_prime += len(self.psi_prime_times[link][k]) + 1
							self.Lambda[link][k] -= mu / (mu+phi) * (self.psi[link][k] - ni)
							self.Lambda[link][k] -= mu_prime / (mu_prime+phi_prime) * (self.psi_prime[link][k] - nj_prime)
						else:
							self.Lambda[link][k] -= mu / (mu+phi) * phi_cusum[k]
							self.Lambda[link][k] -= mu_prime / (mu_prime+phi_prime) * phi_cusum_prime[k]
			## Update the interactions
			if self.interactions:
				self.Lambda[link] += np.sum(self.gamma[link[0]] * (self.gamma_prime[link[1]] if self.directed else self.gamma[link[1]])) * self.A[link]
				if not self.hawkes_int:
					cumulant_sums = 0.0
				if not self.poisson_int:
					for k in range(self.nij[link]):
						if self.hawkes_int:
							self.Lambda[link][k] -= np.sum((nu * nu_prime) / ((nu+theta) * (nu_prime+theta_prime)) * (self.psi_tilde[link][k] - k))
						else:
							if k > 0:
								cumulant_sums -= np.sum((nu * nu_prime) / ((nu+theta) * (nu_prime+theta_prime)) * (np.exp(-(nu+theta) * (nu_prime+theta_prime) * self.A_diff[link][k-1]) - 1))
								self.Lambda[link][k] += cumulant_sums
	
	## Calculate p-values on training and test set
	def pvalues(self, A_test=None, verbose=False):
		## Extend A if necessary
		if A_test is not None:
			nij_old = copy.deepcopy(self.nij)
			self.nij_old = nij_old
			for link in A_test:
				if link in self.A:
					self.A[link] = np.sort(np.array(list(self.A[link])+list(A_test[link])))
				else:
					self.A[link] = np.sort(np.array(A_test[link]))
				self.nij[link] += len(A_test[link])
				self.Tau[link] = 0 if self.tau_zero else self.A[link][0]
			## Construct again out_nodes and in_nodes
			self.out_nodes = {}
			if self.directed:
				self.in_nodes = {}
			for link in self.A:
				if link[0] in self.out_nodes:
					self.out_nodes[link[0]] += [link[1]]
				else:
					self.out_nodes[link[0]] = [link[1]]
				if not self.directed:
					if link[1] in self.out_nodes:
						self.out_nodes[link[1]] += [link[0]]
					else:
						self.out_nodes[link[1]] = [link[0]]
				else:
					if link[1] in self.in_nodes:
						self.in_nodes[link[1]] += [link[0]]
					else:
						self.in_nodes[link[1]] = [link[0]]
		## Model specification
		if not self.main_effects:
			self.hawkes_me = False
		if not self.interactions:
			self.hawkes_int = False
		## Specify again to calculate the required quantities
		self.specification(main_effects=self.main_effects, interactions=self.interactions, 
					poisson_me=self.poisson_me, poisson_int=self.poisson_int, hawkes_me=self.hawkes_me, hawkes_int=self.hawkes_int, verbose=verbose)
		## Setup likelihood calculations
		self.likelihood_calculation_setup(verbose=verbose)
		## Calculate psi
		if ((self.main_effects and self.hawkes_me) or (self.interactions and self.hawkes_int)) and (not self.poisson_me or not self.poisson_int):
			self.psi_calculation(verbose=verbose,calculate_derivative=False)
		## Obtain the compensator values
		self.compensator()
		## Calculate the p-values
		self.pvals_train = {}
		if A_test is not None:
			self.pvals_test = {}
		for link in self.A:
			if A_test is not None:
				## Initialise the required quantities
				if nij_old[link] > 0:
					self.pvals_train[link] = np.zeros(nij_old[link])
				if self.nij[link] > nij_old[link]:
					self.pvals_test[link] = np.zeros(self.nij[link] - nij_old[link])
				if nij_old[link] > 0:
					self.pvals_train[link][0] = 0 if not self.tau_zero else 1 - exp(-self.Lambda[link][0])
					self.pvals_train[link][1:] = 1 - np.exp(-np.diff(self.Lambda[link][:nij_old[link]]))
					self.pvals_test[link] = 1 - np.exp(-np.diff(self.Lambda[link][(nij_old[link]-1):]))
				else:
					self.pvals_test[link][0] = 0 if not self.tau_zero else 1 - exp(-self.Lambda[link][0])
					self.pvals_test[link][1:] = 1 - np.exp(-np.diff(self.Lambda[link]))
			else:
				self.pvals_train[link] = np.zeros(self.nij[link]) 
				self.pvals_train[link][0] = 0 if not self.tau_zero else 1 - exp(-self.Lambda[link][0])
				self.pvals_train[link][1:] = 1 - np.exp(-np.diff(self.Lambda[link]))
