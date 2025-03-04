import numpy as np
from constants import *
from utility_functions import compute_firing_rates_VM

def basic_approach(time_window, spike_counts, phases):
    return phases[ np.argmax(spike_counts[time_window]) ]

# Population vector approach
def weighted_sum_approach(time_window, spike_counts, phases, pv_lengths):
    weights = spike_counts[time_window]
    assigned_vector = np.array(weights * np.exp(1j * phases))
    pv = np.sum(assigned_vector) / np.sum(weights) # this gives me a complex num; radius and a phase
    pv_lengths[time_window] = np.abs(pv) # certainty <-> magnitude 
    return np.angle(pv) # estimation


# Computes likelihood for each time window
def compute_pos_likelihood(pos_index, spike_counts, firing_rates, phases, scaling=scaling, kappa=kappa, max_firing_rate=max_firing_rate):
    fr_vector = compute_firing_rates_VM(firing_rates[:, pos_index], phases, scaling, kappa, max_firing_rate) # recomputing --- very inefficient
    likelihood = np.sum(spike_counts * np.log(fr_vector) - fr_vector)
    return likelihood

# Binary search for high precision floats
def binary_search(sorted_arr, value, epsilon=1e-7):
    left = 0
    right = len(sorted_arr) - 1
    
    while left <= right:
        mid = (left + right) // 2

        if np.isclose(sorted_arr[mid], value, atol=epsilon):
            return mid
        elif sorted_arr[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# computes the likeliest position for each time window based on spike count
# original firing_rates not fr_sums
def mle_approach(time_window, spike_counts, phases, original_positions, positions, firing_rates):
    og_pos = original_positions[time_window]

    max_likelihood = -np.inf
    best_position = None
    for position in og_pos:
        pos_index = binary_search(positions, position)
        likelihood = compute_pos_likelihood(pos_index, spike_counts[time_window], firing_rates, phases)
        if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_position = position
    
    return best_position



# Kalman Filter (uses PV magnitude to adjust R)
def kalman_filter_update(x_prev, P_prev, y, H, R_base, pv_magnitude):
    # Adjust the measurement noise covariance based on PV magnitude
    R = R_base / (pv_magnitude + 1e-6)  # Adding a small value to prevent division by zero
    
    # Prediction
    y_hat = H @ x_prev
    S = H @ P_prev @ H.T + R
    K = P_prev @ H.T @ np.linalg.inv(S)  # Kalman Gain
    
    # Update step
    x_new = x_prev + K @ (y - y_hat)
    P_new = (np.eye(len(x_prev)) - K @ H) @ P_prev
    
    return x_new, P_new

# Recursive algorithm
def decode_positions_KF(decoded_positions, pv_magnitudes, spike_counts, phases, H, R_base, A, Q, x_0, P_0):
    num_time_windows = len(decoded_positions)
    x = x_0  # Initial state estimate
    P = P_0  # Initial covariance estimate
    
    for time_window in range(num_time_windows):
        # Get the PV estimation and magnitude
        estimation = pop_vector_approach(time_window, spike_counts, phases, pv_magnitudes)
        pv_magnitude = pv_magnitudes[time_window]
        
        # Kalman Filter update using PV magnitude to adjust measurement noise
        x, P = kalman_update(x, P, estimation, H, R_base, pv_magnitude)
        
        decoded_positions[time_window] = x[0]  # Store the decoded head direction
        
        # Prediction step
        x = A @ x
        P = A @ P @ A.T + Q
    
    return decoded_positions