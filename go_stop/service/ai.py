"""
ai.py

AI agent.
"""

from typing import List, Tuple

import torch
from torch import Tensor

from ..models.action import NUM_ACTIONS, all_actions
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
        mask = (
            Tensor(
                [
                    1 if all_actions[i] in game.actions() else 0
                    for i in range(NUM_ACTIONS)
                ],
            )
            == 0
        )
        policy = policy.masked_fill(mask, 0)
        print("policy:", policy)
        policy = policy / torch.sum(policy)

        return policy.squeeze().tolist(), value.squeeze().item()
