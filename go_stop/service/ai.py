"""
ai.py

AI agent.
"""

import torch

from ..models.agent import Agent
from ..train.args import args
from ..train.network import EncoderNet


net: EncoderNet = EncoderNet()
ckpt_path = args.root_dir / f"best_{args.num_hidden_layers}_hidden_layers.pt"

if ckpt_path.is_file():
    net.load_state_dict(torch.load(ckpt_path))

ai = Agent.from_net(net)
