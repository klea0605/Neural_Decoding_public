import numpy as np
from scipy.stats import circmean
import astropy.stats as AS

# Map Cartesian positions to a circle
# TODO: not sure if I should include scaling into this:  positions % (2 * np.pi * scaling) -- NO
def circular_positions(positions, num_positions):
    return positions % (2 * np.pi)

#------------------------------------------------------------------

# Circular mean of radian positions
# Radians if not set otherwise
# circ_mean(array_of_radians)
def circ_mean(circ_pos, deg=False, axis=None):
    # n = circ_pos.shape[axis] if axis is not None else len(circ_pos)
    return np.angle(np.mean(np.exp(1j*circ_pos), axis=axis), deg=deg)
#------------------------------------------------------------------

# circ_normalize(circ_difference_two(estimate, initial_head_direction))

def circ_normalize(angle):
    return angle % (2 * np.pi)

#------------------------------------------------------------------

def circ_difference_two(pos1, pos2):
    diff = (pos1-pos2) % 2*np.pi
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff

# 
def circ_difference_arrays(arr1, arr2):
    diff = (arr1 - arr2) % (2 * np.pi)
    diff[diff > np.pi] -= 2 * np.pi  # filtering 
    return diff

#------------------------------------------------------------------