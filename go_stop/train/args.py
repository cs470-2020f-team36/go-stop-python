"""
args.py

Gather the arguments and hyperparameters.
"""

import math
from pathlib import Path

from easydict import EasyDict as edict

from go_stop.models.action import NUM_ACTIONS
from go_stop.train.encoder import DIM_ENCODED_GAME


args = edict()

# Manual seed for reproducibility
args.seed = 470

# Hyperparameters for training the neural network
args.batch_size = 32
args.ninp = DIM_ENCODED_GAME
args.nhid = 256
args.nout = NUM_ACTIONS
args.learning_rate = 0.5
args.lr_decrease_rate = 0.92

# Hyperparameters for MCTS
args.c_puct = 1
args.mcts_search_per_simul = 30
args.num_episodes_per_evolvution = 12
args.max_evolution = 100

# The size of a sample from the observation
args.similar_games = lambda t: max(math.floor((8 * 2 ** ((20 - t) / 8))), 4)

# Temperature parameter: tau = lambda t: 1 if t < args.tau_threshold else args.infinitesimal_tau
args.tau_threshold = 10
args.infinitesimal_tau = 0.05

# Parameters of Dirichlet noises during MCTS
args.epsilon = 0.25
args.alpha = 1

# Create directory name.
args.root = "data"
args.root_dir = Path(args.root)
args.root_dir.mkdir(parents=True, exist_ok=True)
