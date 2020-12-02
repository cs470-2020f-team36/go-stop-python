"""
ai.py

AI agent.
"""

from typing import List, Tuple

import torch

from ..models.agent import Agent
from ..models.game import Game
from ..models.player import Player
from ..train.args import args
from ..train.encoder import encode_game
from ..train.network import EncoderNet


net: EncoderNet = EncoderNet()
ckpt_path = args.root_dir / "best.pt"

if ckpt_path.is_file():
    net.load_state_dict(torch.load(ckpt_path))

ai = Agent.from_net(net)


def estimate(game: Game, player: Player) -> Tuple[List[float], float]:
    """Return the estimation by the neural network."""
    encoded_game = encode_game(game, player)
    net.eval()
    with torch.no_grad():
        policy, value = net(encoded_game.unsqueeze(0))

        return policy.squeeze().tolist(), value.squeeze().item()
