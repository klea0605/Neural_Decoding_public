import numpy as np
from numpy.linalg import norm
from scipy.special import iv
from utility_functions import *
from constants import *

# ------------------------------------------------------------

def extrapolate_HD(pos, n):
    return 2*pos[n] - pos[n-1]

# ------------------------------------------------------------

# This function needs to be called for every time point (1000)
def compute_posterior_spike_count(observed_spike_counts, phases, x_values):
    num_neurons = len(phases)
    num_positions = len(x_values)
    posterior = np.ones_like(x_values)
    fr_x_vals = np.zeros((num_neurons, num_positions))
    
    # Von Mises TCs for optional positions in A time_window
    for pos in range(num_positions):
        fr_x_vals[:, pos] = compute_firing_rates_VM(x_values[pos], phases)
        
    
    for i in range(num_neurons):
        posterior *= poisson.pmf(observed_spike_counts[i], fr_x_vals[i])
    
    posterior /= (np.sum(posterior) + 1e-3) # Normalize
    return posterior

# ------------------------------------------------------------

def compute_posterior_extrapolation(posterior_t, posterior_t_minus_dt, x_values):
    
    num_positions = len(x_values)
    posterior_t_plus_dt = np.zeros(num_positions)
    
    # Convolving in the context of VM distribution
    for i, x in enumerate(x_values):
        for j, y in enumerate(x_values):
            index_t = int(((x - 2*y) % (2*np.pi)) / (2*np.pi) * num_positions) # Index corresponding to (x - 2y)
            index_t_minus_dt = j 
            # Compute the convolution using the Bessel function
            posterior_t_plus_dt[i] += (posterior_t[index_t] * posterior_t_minus_dt[index_t_minus_dt] * iv(0, kappa * np.cos(x - 2*y)))
    
    posterior_t_plus_dt /= (np.sum(posterior_t_plus_dt) + 1e-3)
    
    return np.array(posterior_t_plus_dt)

# ------------------------------------------------------------
