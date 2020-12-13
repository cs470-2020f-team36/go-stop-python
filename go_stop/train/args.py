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

# Control the reward function
args.eval_win_weight = 0

# Manual seed for reproducibility
args.seed = 470

# Hyperparameters for training the neural network
args.batch_size = 256
args.ninp = DIM_ENCODED_GAME
args.nhid = 256
args.nout = NUM_ACTIONS
args.learning_rate = 1e-3
args.lr_decrease_rate = 0.96
args.num_hidden_layers = 6

# Hyperparameters for MCTS
args.c_puct = 4
args.mcts_search_per_simul = 50
args.num_episodes_per_evolvution = 35
args.max_evolution = 1000
args.mcts_reward_weight = 0.7
args.replay_buffer_size = lambda c: 500 + c * 400

# The size of a sample from the observation
args.num_similar_games = lambda t: max(math.floor(4 * (2 ** ((20 - t) / 7))), 3)
args.num_evaluation_games = 5000

# Temperature parameter: tau = lambda t: 1 if t < args.tau_threshold else args.infinitesimal_tau
args.tau_threshold = 20
args.infinitesimal_tau = 0.01

# Parameters of Dirichlet noises during MCTS
args.epsilon = 0.25
args.alpha = 1

# Create directory name.
args.root = "data"
args.root_dir = Path(args.root)
args.root_dir.mkdir(parents=True, exist_ok=True)
