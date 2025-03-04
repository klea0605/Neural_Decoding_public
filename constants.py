num_neurons = 10
num_positions = 1000
dt = 100*1.e-3 # (s)
dt_ms = dt*1000
T_ms = dt_ms*num_positions
sc_time_step = 1 #TODO: this should allow us to have ~ 2 spikes for position
tau = 0.3  # Time constant for Ornstein-Uhlenbeck process
mu = 0.0  # Mean of Ornstein-Uhlenbeck process
sigma = 0.8  # Standard deviation of Ornstein-Uhlenbeck process (0.5 better)
scaling = 1.0  # Grid cell spacing (not used in 1D ring model)
kappa = 2.0  # Concentration parameter for von Mises distribution
max_firing_rate = 50  # lowers the spike count
dropout_proba = 0.3
dropout = False
pv_decoding_threshold = 0.4

# T = 15  time window for spike measuring (ms) -- 5s
# dt = T/num_positions  # Time step for generating positions (ms)
