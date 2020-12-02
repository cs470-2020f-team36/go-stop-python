"""
ai.py

AI agent.
"""

import torch

from ..train.args import args
from ..train.network import EncoderNet
from ..models.agent import Agent


net: EncoderNet = EncoderNet()
ckpt_path = args.root_dir / "best.pt"

if ckpt_path.is_file():
    net.load_state_dict(torch.load(ckpt_path))

ai = Agent.from_net(net)
