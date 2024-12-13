"""Configurations for the MPC controller."""

import torch
import numpy as np
from enum import Enum, auto

class DynamicsModel(Enum):
    SINGLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR = auto()

# Liveness parameters.
liveliness = True
liveness_threshold = 0.8
plot_rate = 1
plot_live = True
plot_live_pause = False
plot_arrows = False
plot_end = False
save_data_path = None
# save_data_path = 'doorway_train_data_with_liveness_1_faster.json'

# dynamics = DynamicsModel.SINGLE_INTEGRATOR
dynamics = DynamicsModel.DOUBLE_INTEGRATOR
mpc_p0_faster = True

if dynamics == DynamicsModel.SINGLE_INTEGRATOR:
    num_states = 3 # (x, y, theta)
    num_controls = 2 # (v, omega)
else:
    num_states = 4 # (x, y, theta, v)
    num_controls = 2 # (a, omega)

n = 2                                      # Number of agents
runtime = 22.0                             # Total runtime [s]
# runtime = 5.0
sim_ts = 0.2                                # Simulation Sampling time [s]
MPC_Ts = 0.1                                   # MPC Sampling time [s]
T_horizon = 4                              # Prediction horizon time steps
sim_steps = int(runtime / sim_ts)              # Number of iteration steps for each agent

obstacle_avoidance = True
obs_gamma = 0.2                            # CBF parameter in [0,1]
liveliness_gamma = 0.3                     # CBF parameter in [0,1]
# safety_dist = 0.00                         # Safety distance
# agent_radius = 0.01                         # Robot radius (for obstacle avoidance)
safety_dist = 0.03                         # Safety distance
agent_radius = 0.1                         # Robot radius (for obstacle avoidance)
zeta = 2.0

# Actuator limits
v_limit = 0.30                             # Linear velocity limit
omega_limit = 0.5                          # Angular velocity limit
accel_limit = 0.1

# ------------------------------------------------------------------------------
COST_MATRICES = {
    DynamicsModel.SINGLE_INTEGRATOR: {
        "Q": np.diag([15, 15, 0.005]),  # State cost matrix DOORWAY
        # "Q": np.diag([100, 100, 11]), # State cost matrix INTERSECTION
        "R": np.array([3, 1.5]),                  # Controls cost matrix
    },
    DynamicsModel.DOUBLE_INTEGRATOR: {
        "Q": np.diag([15, 15, 0.005, 10.0]),  # State cost matrix DOORWAY
        # "Q": np.diag([100, 100, 11, 3]), # State cost matrix INTERSECTION
        "R": np.array([0.5, 10.0]),                  # Controls cost matrix
    }
}

# Training parameters.
use_barriernet = True
agent_to_train = 1
# train_data_path = 'doorway_train_data_no_liveness.json'
train_data_paths = ['doorway_train_data_with_liveness_0_faster.json', 'doorway_train_data_with_liveness_1_faster.json']
train_batch_size = 24

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

''' In order to use MPS, change all .double() to float() '''
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )


learning_rate = 1e-3
epochs = 10
nHidden1 = 128
nHidden21 = 32
nHidden22 = 32
# l = liveness, nl = no liveness
# saf = trained on both slow and fast variations.
saveprefix = f'model_l_saf_{agent_to_train}'
