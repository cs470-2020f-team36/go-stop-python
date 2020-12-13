"""
elo.py

Calculate Elo rating (among two agents.)
Note that the result is highly fluctuating due to the randomness of the random agent,
and also the result does not tell the strength of the agent because the elo rating
does NOT reflect the absolute value of the scores, but only the sign of scores.

This file is just for fun :)
"""

from __future__ import annotations

import torch

from go_stop.models.agent import Agent
from go_stop.train.args import args
from go_stop.train.network import EncoderNet
from go_stop.utils.elo import elo

if __name__ == "__main__":
    net_3 = EncoderNet(3)
    ckpt_path_3 = args.root_dir / "best_3_hidden_layers.pt"
    if ckpt_path_3.is_file():
        net_3.load_state_dict(torch.load(ckpt_path_3))

    net_6 = EncoderNet(6)
    ckpt_path_6 = args.root_dir / "best_6_hidden_layers.pt"
    if ckpt_path_6.is_file():
        net_6.load_state_dict(torch.load(ckpt_path_6))

    random_agent = Agent.random()
    print("AGS-3 (vs random):", elo(Agent.from_net(net_3), random_agent))
    print("AGS-6 (vs random):", elo(Agent.from_net(net_6), random_agent))
    print("AGS-6 (vs AGS-3) :", elo(Agent.from_net(net_6), Agent.from_net(net_3)))
