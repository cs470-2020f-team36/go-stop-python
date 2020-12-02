"""
agent.py

Implement an agent class.
"""

from abc import ABC
import random
from typing import Literal, Optional

import torch
from torch import Tensor

from .action import NUM_ACTIONS, all_actions
from .game import Game
from ..train.encoder import encode_game
from ..train.network import EncoderNet

Kind = Literal["test", "random", "net"]


class Agent(ABC):
    """
    Agent class.

    test: CLI agent
    random: random agent
    net: agent from the neural network
    """

    def __init__(self, kind, net: Optional[EncoderNet] = None):
        # Game -> Action
        self.kind = kind
        self.net = net

    def query(self, game: Game):
        """Query an action to the agent with given game"""

        if self.kind == "test":
            return game.actions()[int(input())]

        if self.kind == "random":
            return random.choice(game.actions())

        if self.kind == "net":
            net = self.net
            net.eval()
            with torch.no_grad():
                encoded_game = (
                    torch.from_numpy(encode_game(game, game.state.player))
                    .unsqueeze(0)
                    .float()
                )
                mask = (
                    Tensor(
                        [
                            1 if all_actions[i] in game.actions() else 0
                            for i in range(NUM_ACTIONS)
                        ],
                    )
                    == 0
                )

                action_index = (
                    net(encoded_game)[0]
                    .squeeze()
                    .masked_fill(mask, 0)
                    .argmax()
                    .item()
                )

                action = all_actions[action_index]
                if action not in game.actions():
                    action = random.choice(game.actions())
                return action

    @staticmethod
    def test():
        """Make a CLI agent."""
        return Agent("test")

    @staticmethod
    def random():
        """Make a random agent."""
        return Agent("random")

    @staticmethod
    def from_net(net):
        """Make a neural network based agent."""
        return Agent("net", net)
