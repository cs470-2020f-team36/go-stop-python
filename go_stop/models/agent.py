"""
agent.py

Implement an agent class.
"""

from abc import ABC
import random
from typing import Literal, Optional

from numpy.random import choice
import torch
from torch import Tensor

from .action import NUM_ACTIONS, ALL_ACTIONS
from .game import Game
from ..train.args import args
from ..train.encoder import encode_game
from ..train.network import EncoderNet
from ..utils.exp import mean_exp

Kind = Literal["test", "random", "net"]


class Agent(ABC):
    """
    Agent class.

    test: CLI agent
    random: random agent
    net: agent from the neural network
    """

    def __init__(self, kind: Kind, net: Optional[EncoderNet] = None):
        self.kind = kind
        self.net = net

    def query(self, game: Game):
        """Query an action to the agent with given game."""

        if self.kind == "test":
            return game.actions()[int(input())]

        if self.kind == "random":
            return random.choice(game.actions())

        assert self.kind == "net"

        estimation = self.estimate(game)

        if estimation is None:
            action = random.choice(game.actions())
            return action

        policy, _ = estimation
        policy = mean_exp(torch.from_numpy(policy), 1 / args.infinitesimal_tau).numpy()
        try:
            action = choice(ALL_ACTIONS, size=1, p=policy)[0]
        except ValueError:
            action = random.choice(game.actions())

        return action

    def estimate(self, game: Game):
        """Return an estimate of the policy and the expected value given a state."""
        if game.actions() == []:
            return None

        if self.kind == "net":
            net = self.net
            net.eval()
            with torch.no_grad():
                encoded_game = encode_game(game, game.state.player).unsqueeze(0).float()
                mask = (
                    Tensor(
                        [
                            1 if ALL_ACTIONS[i] in game.actions() else 0
                            for i in range(NUM_ACTIONS)
                        ],
                    )
                    == 0
                )

                policy_, value = net(encoded_game)
                policy = policy_.squeeze().masked_fill(mask, 0)

                print(policy, policy.sum().item())

                if policy.sum().item() == 0:
                    # Perform a random action when every action has estimated probability 0
                    policy = torch.ones((NUM_ACTIONS,)).masked_fill(mask, 0)
                    policy = policy / policy.sum()
                else:
                    policy = policy / policy.sum()

                policy = policy.numpy()

                value = value.squeeze().item()

                return policy, value

        return None

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
