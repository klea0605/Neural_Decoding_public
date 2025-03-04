from matplotlib import pyplot as plt
import numpy as np
from circular_arithmetics import *
from constants import *
from numpy.linalg import norm



def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


# Plot the OU process
# Important: function call with uncirculated positions
def plt_positions_uncirculated(positions, num_positions, dt, title:str):
    plt.figure(figsize=(8, 10))
    time_steps = np.arange(num_positions)
    # time_steps = np.arange(num_positions)*dt
    plt.plot(time_steps, positions, label='OU Process')
    plt.title(title)
    # plt.xlabel('Time (s)')
    plt.xlabel('Time (indices)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(False)
    plt.show()
#----------------------------------------------------------------------------------------------------

# Plot Phases: --------------------------------------------------------------------------------------
def plt_equidist_phases(x_coords, y_coords, angles, rad=True):
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='red', marker='o', label='Preferred spatial phases')
    
    # Circle
    plt.plot(np.cos(np.linspace(0, 2 * np.pi, 1000)), np.sin(np.linspace(0, 2 * np.pi, 1000)), linestyle='solid', color='black')

    format_txt = 'Radians' if rad is True else 'Angles'

    def format_angle(angle):
        if rad:
            return f"{angle:.2f}"  # Show radians as float numbers
        else:
            if angle == 0:
                return "0"
            elif angle == np.pi:
                return "π"
            elif angle == 2 * np.pi:
                return "2π"
            else:
                fraction = angle / np.pi
                return f"{fraction:.1f}π"

    for i, angle in enumerate(angles):
        plt.text(x_coords[i], y_coords[i], format_angle(angle), fontsize=12, ha='right')
    
    plt.title(f'Evenly Spaced Grid Cells - {format_txt}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


#----------------------------------------------------------------------------------------------------

# Plot circulated positions
def plt_circulated_pos(circ_positions, phases):
    plt.figure(figsize=(13, 13))
    ax = plt.subplot(2, 1, 1, projection='polar')

    # Plot the circular positions
    ax.scatter(circ_positions, np.ones_like(circ_positions), s=10, marker = 'x' , label='Head Direction Trajectory')

    # Plot the preferred head directions (red circles)
    ax.scatter(phases, np.ones_like(phases), s=20, color='red', marker='o', label='Preferred Neural Head Directions - Phases')

    ax.set_title('Circular plot of Generated Head Directions', fontsize = 15)
    ax.set_yticklabels([])  # Hide the radial labels
    ax.set_xticks(phases)  # Set angular ticks
    ax.set_xticklabels(np.round(phases, 2))
    ax.legend(loc='center', fontsize = 12)
#----------------------------------------------------------------------------------------------------


# Plot firing rates
# TODO: add a marker for every position to see if the linear jump is due to positions
def plt_firing_rates_VM(positions, num_neurons, firing_rates, num_positions, title : str):
    plt.figure(figsize=(10, 6))
    for neuron in range(num_neurons):
        plt.plot(positions, firing_rates[neuron, :], label=f'Neuron {neuron+1}')
    plt.title(f'Firing Rates of Neurons (von Mises Distribution) --> {title}')
    plt.xlabel(f'Position (cartesian)' if title.lower() == 'cartesian' else 'Position (radians)')
    plt.ylabel('Firing Rate')
    plt.xticks([])
    plt.legend(loc = 'lower left')
    plt.show()
#----------------------------------------------------------------------------------------------------

# TODO: this is a cpy-paste; doesn't work well - neuron index is useless
def plt_spike_trains(spike_counts):
    plt.subplot(2, 1, 2)
    plt.imshow(spike_counts, aspect='auto', cmap='binary', interpolation='none')
    plt.title('Spike Trains')
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')

#---------------------------------------------------------------------------------------------------------------------------------------

def plt_decoded_positions_cmp_mean_og(tw_original_positions, decoded_positions, circ_positions, sc_time_step, num_time_windows_limit, tw_bounds:bool, approach_name:str):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')

    num_positions = len(circ_positions)

    # Plotting 10 time windows
    for time_window in range(num_time_windows_limit):
        # mean_original_pos_tw = np.mean(tw_original_positions[time_window])
        mean_original_pos_tw = circ_normalize(circ_mean(tw_original_positions[time_window]))

        # Plot mean original position
        ax.plot(mean_original_pos_tw, 1, 'bo', label='Mean Original Location' if time_window == 0 else "")

        # Plot decoded position
        ax.plot(decoded_positions[time_window], 1, 'r+', label='Decoded Location' if time_window == 0 else "")

        # Draw boundary lines for time windows
        if tw_bounds:
            start_pos = circ_positions[time_window * sc_time_step]
            # end_pos = circ_positions[min((time_window + 1) * sc_time_step - 1, num_positions - 1)]
            end_pos = circ_positions[min((time_window + 1) * sc_time_step-1, num_positions - 1)]
            ax.plot([start_pos, start_pos], [0, 1], 'y-', linewidth=0.5, label = 'TW beginning' if time_window == 0 else "")
            ax.plot([end_pos, end_pos], [0, 1], 'k-', linewidth=0.5, label = 'TW end' if time_window == 0 else "")

    # Add legend
    plt.legend(loc='upper right')
    ax.set_title(f'{approach_name} approach')
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

def plt_square_diff(mse, squared_diff, approach_name:str, threshold_meth = 'min_max', k = None, skip_ticks_step=10):

    if threshold_meth == 'min_max':
        treshold = (max(squared_diff) - min(squared_diff))/2
    elif threshold_meth == 'mean_std':
        threshold = np.mean(squared_diff) + k*np.std(squared_diff)
    else:
        raise ValueError(f"Invalid threshold method: {threshold_meth}")

    plt.figure(figsize=(15, 8))
    tw_range = range(len(squared_diff))
    plt.plot(tw_range, squared_diff, 'o-', label = f'{approach_name} approach - Square Differences')
    plt.plot([], [], " ", label = f'{approach_name} approach MSE = {np.round(mse, 3)}, SSD = {np.round(sum(squared_diff), 3)} ')
    plt.axhline(y=threshold, color='green', linestyle='-')

    plt.xticks(tw_range[::skip_ticks_step])
    plt.xlabel('Time windows')
    plt.ylabel('Square difference')
    plt.legend(loc = 'upper left', fontsize = 'large')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

def plt_real_vs_decoded_trajectory(decoded_positions, tw_original_positions, approach_name, skip_ticks_step = 10, sc_time_step=sc_time_step):
    num_time_windows = len(decoded_positions)

    # Normalize positions
    mean_og_pos = circ_normalize(circ_mean(tw_original_positions, axis=1))
    decoded_pos = circ_normalize(decoded_positions)
    
    # Time axis
    time_axis = np.arange(0, num_time_windows * sc_time_step, sc_time_step)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(time_axis, mean_og_pos, 'o-', label='Real Trajectory (Circular Mean)', color='blue')
    plt.plot(time_axis, decoded_pos, 'x-', label=f'Decoded Trajectory ({approach_name} approach)', color='red')
    
    plt.xticks(time_axis[::skip_ticks_step])
    plt.xlabel('Time (indices)')
    plt.ylabel('Angle (radians)')
    plt.legend(loc='upper left', fontsize='large')
    plt.title('Comparison of Real and Decoded Trajectories')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------

# !!!! This does not work well !!!
# 2 problems: 
# 1) negative mean ??
# 2) treating close radians as really far away (plot)
def plt_circular_distance(decoded_positions, tw_original_positions, approach_name: str, skip_ticks_step=10):
    num_time_windows = len(tw_original_positions)
    mean_og_pos = circ_mean(tw_original_positions, axis=1)
    circ_differences = np.array([circ_difference_two(decoded_positions[tw], mean_og_pos[tw]) for tw in range(num_time_windows)])

    plt.figure(figsize=(15, 8))
    tw_range = range(num_time_windows)
    plt.plot(tw_range, circ_differences, 'o-', label=f'{approach_name} approach - Circular Distances')
    plt.plot([], [], " ", label=f'{approach_name} approach Mean CD = {np.round(circ_mean(circ_differences), 3)}')
    plt.xticks(tw_range[::skip_ticks_step])
    plt.xlabel('Time windows')
    plt.ylabel('Circular Distances')
    plt.legend(loc='upper left', fontsize='large')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------

def calculate_correlation_per_tw(square_diff, stat_values):
    num_tw = len(square_diff)
    correlation_coeffs = np.zeros(num_tw)
    for tw in range(num_tw):
        if tw < num_tw - 1:
            correlation_coeffs[tw], _ = pearsonr([square_diff[tw], square_diff[tw + 1]], [stat_values[tw], stat_values[tw + 1]])
        else:
            correlation_coeffs[tw], _ = pearsonr([square_diff[tw - 1], square_diff[tw]], [stat_values[tw - 1], stat_values[tw]])
    return correlation_coeffs

def plot_spike_count_statistics_vs_squared_diff(square_diff, spike_counts, sc_stat='all', threshold_meth = 'min_max', k = None ,skip_ticks_step=10):
    # Calculate the statistics
    spike_count_stats = {
        'mean': np.mean(spike_counts, axis=1),
        'median': np.median(spike_counts, axis=1),
        'var': np.var(spike_counts, axis=1)
    }

    if threshold_meth == 'min_max':
        sqd_threshold = (max(square_diff) - min(square_diff))/2
    elif threshold_meth == 'mean_std':
        sqd_threshold = np.mean(square_diff) + k*np.std(square_diff)
    else:
        raise ValueError(f"Invalid threshold method: {threshold_meth}")

    # Identify indices with high squared differences
    high_sq_diff_indices = np.where(square_diff > sqd_threshold)[0]  # taking the preceding tw

    # Plot the statistics vs squared differences
    plt.figure(figsize=(20, 10))
    tw_range = range(len(square_diff))

    def plot_statistic(stat_name, subplot_position):
        plt.subplot(2, 2, subplot_position)
        plt.scatter(tw_range, square_diff, c='blue', label='Squared Differences', alpha=0.6) # err against time
        plt.scatter(tw_range, spike_count_stats[stat_name], c='green', label=f'{stat_name.capitalize()} Spike Counts', alpha=0.6) # 
        plt.scatter(high_sq_diff_indices, square_diff[high_sq_diff_indices], c='red', label='High Squared Differences')
        for i in high_sq_diff_indices:
            plt.plot([i, i], [square_diff[i], spike_count_stats[stat_name][i]], 'r--')
        plt.xticks(tw_range[::skip_ticks_step])
        plt.xlabel('Time windows')
        plt.ylabel('Values')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.title(f'{stat_name.capitalize()} Spike Counts vs Squared Differences')

    if sc_stat == 'all':
        plot_statistic('mean', 1)
        plot_statistic('median', 2)
        plot_statistic('var', 3)
    else:
        plt.figure(figsize=(20, 8))
        plot_statistic(sc_stat, 1)

    plt.tight_layout()
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

# Easy scatter 

def plt_scatter_sqd_sc(spike_counts, squared_diff, approach_name):
    spike_sums = np.sum(spike_counts, axis = 1) # x-axis
    plt.figure(figsize=(15, 5))
    plt.scatter(spike_sums, squared_diff)
    plt.plot([], [], label = f'Sq Differences VS ∑(Spike Counts) -- {approach_name}')
    plt.xticks(spike_sums)
    plt.xlabel('Spike sums', fontsize=12)
    plt.ylabel('Square differences')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------

def plot_binned_polar_population_vector(time_window, spike_counts, decoded_positions, pv_lengths, tw_original_positions, phases, text):
    
    # Extract assigned vectors for the given time window
    num_neurons = spike_counts[time_window].size
    mean_og_pos = circ_normalize(circ_mean(tw_original_positions[time_window]))
    max_len = np.max(pv_lengths)
    min_len = np.min(pv_lengths)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Pop vector
    estimation = decoded_positions[time_window]  # angle
    length = pv_lengths[time_window]  # magnitude

    # Ensure the polar plot is scaled properly
    max_bin_height = np.max(spike_counts[time_window])
    ax.set_ylim(0, max_bin_height)
    ax.set_rmax(max_bin_height)  # Set rmax to the maximum spike count

    # Pct of certainty (based on the max certainty we can get):
    certainty_pct = np.round(length * 100 / max_len, 2)
    
    # Plotting
    # Bins - properly scaled to the spike counts
    ax.bar(phases, spike_counts[time_window], width=2*np.pi/num_neurons, color='xkcd:powder blue', alpha=0.7, edgecolor='k')

    # Scale the arrow length based on the certainty_pct and max_len
    arrow_length = (length / max_len) * max_bin_height
    ax.arrow(0, 0, estimation, arrow_length, color='red', linewidth=2, 
             label=f'PV\nEstimation: {circ_normalize(estimation):.2f}\nCertainty: {certainty_pct} / 100 %',
             head_width=0.05, head_length=0.1)

    # Plot the original mean position as a blue dot
    ax.plot(mean_og_pos, max_bin_height, color='xkcd:vibrant blue', marker='o', label=f'Original Position: {mean_og_pos:.2f}')

    # Remove redundant scale lines
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, num_neurons, endpoint=False))
    ax.set_xticklabels([f'{x:.2f}' for x in phases])

    plt.title(f'Binned Population Vector Polar Plot \n Time window: {time_window} \n {text}')
    plt.legend(loc='upper right')

    plt.show()


    #-------------------------------------------------------------------------------------------------------------------------------------------


    # !!! Use circular arithmetics

    # Cosine similarity
    # Plot: 
    # Legend: cosine similarity num
    # Plot pop_vector magnitudes --> they should be high if cos->1
def plt_pv_cos_sim_vel(estimated_vel, true_vel, pv_magnitudes):
    # Assuming pv_magnitudes is an array of PV magnitudes per time window
    time_windows = np.arange(len(pv_magnitudes))

    plt.figure(figsize=(12, 6))
    plt.plot(time_windows, pv_magnitudes, label=f'PV Magnitude\nCosine Similarity: {np.dot(estimated_vel, true_vel) / (norm(estimated_vel) * norm(true_vel)):.2f}', color='b')
    plt.xlabel('Time Windows')
    plt.ylabel('PV Magnitude')
    plt.title('Velocities Similarity and Certainty of Esitimates')
    plt.legend()
    plt.show()

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # Plot: 
    # Legend: variance between estimated and true positions
    # Plot pop_vector magnitudes --> they should be high if variance is low
def plt_pv_std_pos(estimated_pos, true_pos):
    pass # cosine similarity between two vectors

    #-------------------------------------------------------------------------------------------------------------------------------------------

def PV_plt_real_vs_decoded_trajectory(decoded_positions, original_positions, approach_name, skip_ticks_step = 50, sc_time_step=sc_time_step):
    num_time_windows = len(decoded_positions)

    # Normalize positions
    # mean_og_pos = circ_normalize(circ_mean(tw_original_positions, axis=1))
    # decoded_pos = circ_normalize(decoded_positions)
    
    # Time axis
    time_axis = np.arange(0, num_time_windows * sc_time_step, sc_time_step)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(time_axis, original_positions, 'o-', label='Real Trajectory (Circular Mean)', color='blue')
    plt.plot(time_axis, decoded_positions, 'x-', label=f'Decoded Trajectory ({approach_name} approach)', color='red')
    
    plt.xticks(time_axis[::skip_ticks_step])
    plt.xlabel('Time (indices)')
    plt.ylabel('Angle (radians)')
    plt.legend(loc='upper left', fontsize='large')
    plt.title('Comparison of Real and Decoded Trajectories')
    plt.grid(True)
    plt.tight_layout()
    plt.show()   

#-------------------------------------------------------------------------------------------------------------------------------------------

def PV_plot_trajectory_comparison(real_positions, decoded_positions, num_positions, decoding_colors, pv_decoding_threshold, dropout:bool, dropout_proba = None, dt=dt, title='Real VS Decoded Trajectory'):
    plt.figure(figsize=(8, 10))
    time_steps = np.arange(num_positions) * dt
    
    # Plot the real trajectory (OU Process)
    plt.plot(time_steps, real_positions, label='Real Trajectory (OU Process)', color='blue', linewidth=3)

    # Plot the decoded trajectory segment by segment
    for i in range(1, num_positions):
        plt.plot(time_steps[i-1:i+1], decoded_positions[i-1:i+1], color=decoding_colors[i], marker='x', markersize = 0.7)

    plt.title(title)
    plt.plot([], [], "-", label='Extrapolation method', color='yellow', linewidth=3)
    plt.plot([], [], "-", label='PV method', color='magenta', linewidth=3)
    plt.plot([], [], " ", label=f'Cosine similarity between real and decoded trajectory: {cosine_similarity(decoded_positions, real_positions):.3f}')
    plt.plot([], [], " ", label=f'Extrapolation threshold: {pv_decoding_threshold}')
    plt.plot([], [], " ", label=f'Spike counts dropouts: {dropout}')
    if dropout_proba is not None:
        plt.plot([], [], " ", label=f'Dropout threshold: {dropout_proba}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (Cartesian)')
    plt.legend()
    plt.grid(True)
    plt.show()