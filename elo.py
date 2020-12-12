"""
train.py

Train the neural network.
Note that this project does not use GPU,
as the most time-consuming step in this project is MCTS,
which is difficult to accelerate using GPU.
"""

from __future__ import annotations
import json
import math
import pickle
import random
from typing import Dict, List, Optional, Tuple

import copy
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
import tqdm

from go_stop.models.agent import Agent
from go_stop.models.game import Game
from go_stop.models.action import (
    Action,
    NUM_ACTIONS,
    ALL_ACTIONS,
    get_action_index,
)
from go_stop.train.args import args
from go_stop.train.network import EncoderNet
from go_stop.train.match import match_agents


K_FACTOR = 20


def elo(num_hidden_layers=args.num_hidden_layers):
    """Calculate ELO rating."""
    net = EncoderNet(num_hidden_layers)

    ckpt_path = args.root_dir / "old" / f"best_{num_hidden_layers}_hidden_layers.pt"
    if ckpt_path.is_file():
        net.load_state_dict(torch.load(ckpt_path))

    points = match_agents(Agent.from_net(net), Agent.random(), num_evaluation_games=10000)
    wins = [1 if p > 0 else 0.5 if p == 0 else 0 for p in points]

    rating = [0, 0]

    for win in wins:
        exp_rating = [10 ** (r / 400) for r in rating]
        exp_win_rate_of_first_player = exp_rating[0] / sum(exp_rating)

        rating[0] = rating[0] + K_FACTOR * (win - exp_win_rate_of_first_player)
        rating[1] = rating[1] - K_FACTOR * (win - exp_win_rate_of_first_player)

    # Return the relative Elo score where the random agent has Elo rating 0.
    return rating[0] * 2


if __name__ == "__main__":
    print("AGS-3:", elo(3))
    print("AGS-6:", elo(6))
