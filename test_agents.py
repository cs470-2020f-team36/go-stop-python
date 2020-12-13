"""
test_agents.py

Test agents.
"""

from __future__ import annotations

import numpy as np
import torch

from go_stop.models.agent import Agent
from go_stop.train.args import args
from go_stop.train.match import match_agents
from go_stop.train.network import EncoderNet

if __name__ == "__main__":
    NUM_HIDDEN_LAYERS = 6 # or 3
    net = EncoderNet(NUM_HIDDEN_LAYERS)

    ckpt_path =  args.root_dir / f"best_{NUM_HIDDEN_LAYERS}_hidden_layers.pt"
    model_dict = torch.load(ckpt_path)
    net.load_state_dict(model_dict)

    points = match_agents(Agent.from_net(net), Agent.random())
    print(
        f"non-defeat rate: {len([x for x in points if x >= 0]) / len(points)},",
        f"win_rate: {len([x for x in points if x > 0]) / len(points)},",
        f"mean points: {np.mean(points)},",
        f"std points: {np.std(points)}",
    )
