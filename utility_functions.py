import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson
from constants import max_firing_rate, scaling, kappa, dt, sc_time_step
from plot_functions import plt_equidist_phases
from circular_arithmetics import *

# T = trajectory time (of the experiment basically)
# Time vector: t = np.linspace(0, T, num_steps)


# Generating input positions using the Ornstein Uhlenbeck process:
def get_OU_inputs(x0, v0, num_positions, dt, tau, mu, sigma):
    x = np.zeros(num_positions)
    v = np.zeros(num_positions)
    x[0] = np.round(x0, 8)
    v[0] = v0
    for t in range(1, num_positions):
        dv = dt / tau * (mu - v[t-1]) + sigma * np.sqrt(dt/tau) * np.random.randn()
        v[t] = v[t-1] + dv
        dx = v[t] * dt
        x[t] = x[t-1] + dx  # len(x) == 1000
    return np.round(x, 8), np.round(v, 3)
 #------------------------------------------------------------------   

# Equidistant phases
def get_evenly_spaced_phases(num_neurons:int, plot: bool, rad: bool):
    angles = np.linspace(0, 2*np.pi, num_neurons, endpoint = False)
    # Convert to Cartesian
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)
    if plot:
        plt_equidist_phases(x_coords, y_coords, angles, rad=rad)
    return angles
#------------------------------------------------------------------

# Generate firing rates using modified von Mises distribution
# !! Positions argument must be circular 
def compute_firing_rates_VM(circular_positions, phases, scaling=scaling, kappa=kappa, max_firing_rate=max_firing_rate):
    rates = max_firing_rate*np.exp( kappa * ( np.cos ((circular_positions - phases)/scaling) - 1))
    return rates
#------------------------------------------------------------------

# Inputs -> Rates -> Spikes
# Counting over ALL generating positions you get the spike count vector

# Generate Poisson distributed spikes
def generate_spikes_Poiss(firing_rates, sc_time_window):
    return poisson.rvs(firing_rates * sc_time_window)
#------------------------------------------------------------------

# Decoding positions ------------------------------------------------------------------------------------------------------------------------------

def help_func(ax, circ_positions, color, title=""):
        ax.scatter(circ_positions, np.ones_like(circ_positions), s=10, c=color)
        ax.set_title(f'Circular positions {title}')
        ax.set_yticklabels([])  # Hide the radial labels
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=8, endpoint=False))  # Set angular ticks
        ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])

def test_decoded_pos_rnd_index(tw_original_positions, decoded_positions, approach_name: str):

    ntw = len(decoded_positions)
    rnd_tw = int(np.random.uniform(ntw))
    print(f'Random chosen tw: {rnd_tw}')
    
    mean_og_pos = circ_normalize(circ_mean(tw_original_positions[rnd_tw]))  # keep radians
    decoded_pos = circ_normalize(decoded_positions[rnd_tw])
    
    print(f'Circ Mean original position for given tw (normalized radians): {mean_og_pos}')
    print(f'Decoded position (normalized radians): {decoded_pos}')

    fig, plots = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})  # Create a figure with 1 row and 2 columns

    help_func(plots[0], mean_og_pos, color='b', title=f'Mean Original Position\n({approach_name} approach)') 
    help_func(plots[1], decoded_pos, color='r', title=f'Decoded Position\n({approach_name} approach)')

    plt.tight_layout()
    plt.show()

    return rnd_tw
    #------------------------------------------------------------------
    #------------------------------------------------------------------


def test_decoded_pos_rnd_index_PV(tw_og_cartesian, decoded_pos_unwraped, approach_name: str):

    ntw = len(decoded_pos_unwraped)
    rnd_tw = int(np.random.uniform(ntw))
    print(f'Random chosen tw: {rnd_tw}')

    mean_og_pos = circ_normalize(np.mean(tw_og_cartesian[rnd_tw]))
    decoded_pos = circ_normalize(decoded_pos_unwraped[rnd_tw])

    print(f'Circ Mean original position for given tw (normalized radians): {mean_og_pos:.3f}')
    print(f'Decoded position (normalized radians): {decoded_pos:.3f}')

    fig, plots = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})  # Create a figure with 1 row and 2 columns

    help_func(plots[0], mean_og_pos, color='b', title=f'Mean Original Position\n({approach_name} approach)') 
    help_func(plots[1], decoded_pos, color='r', title=f'Decoded Position\n({approach_name} approach)')

    plt.tight_layout()
    plt.show()

    return rnd_tw

#------------------------------------------------------------------

def decode_positions(decoded_positions, approach_func, *args, **kwargs):
    num_time_windows = len(decoded_positions)
    for time_window in range(num_time_windows):
       decoded_positions[time_window] = approach_func(time_window, *args, **kwargs)

# Example usage: decode_positions(decoded_positions, weighted_sum_approach, spike_counts, phases)

#------------------------------------------------------------------

# returns mse: number over all decoded positions  and   sq_diff: matrix of shape (num_time_windows, 1)
def MSE_decoded_original_pos(decoded_positions, original_positions):
    mean_og_pos = circ_normalize(circ_mean(original_positions, axis=1)) # keep radians
    square_diff = circ_difference_arrays(decoded_positions, mean_og_pos)**2 
    mse = np.mean(square_diff)
    # mse = circ_mean(square_diff)
    return mse, square_diff # sq_diff for plotting
#------------------------------------------------------------------


def extract_significant_sd(square_diff, spike_counts, threshold_factor=0.5):

    threshold = (max(square_diff) - min(square_diff)) * threshold_factor
    num_tw = len(spike_counts)

    significant_jumps = []

    for tw in range(num_tw - 1):
        if square_diff[tw + 1] - square_diff[tw] > threshold:
            significant_jumps.append({
                'tw_index': tw+1,
                'median_spike_count': np.median(spike_counts[tw+1])
            })
    
    return significant_jumps



#------------------------------------------------------------------

# POPULATION VECTOR SPECIFICLY

# ??? do I need OU dt or do i just look at different time windows ?
# ??? I think I should use abs;
# TODO: print velocities to see if they are various ±

def pv_estim_vels(decoded_positions_unwraped, dt=dt, sc_time_step=sc_time_step):
    num_time_windows = len(decoded_positions_unwraped)
    est_vels = np.array([(decoded_positions_unwraped[i] - decoded_positions_unwraped[i-1]) / (sc_time_step*dt) for i in range(1, num_time_windows)])
    return est_vels

#------------------------------------------------------------------

