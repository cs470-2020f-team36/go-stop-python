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
from go_stop.train.match import match_agents


K_FACTOR = 20


def elo(agent_a: Agent, agent_b: Agent):
    """Return the difference of Elo ratings between `agent_a` and `agent_b`."""
    points = match_agents(agent_a, agent_b, num_evaluation_games=10000)
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
    net_3 = EncoderNet(3)
    ckpt_path_3 = args.root_dir / "best_3_hidden_layers.pt"
    if ckpt_path_3.is_file():
        net_3.load_state_dict(torch.load(ckpt_path_3))

    net_6 = EncoderNet(6)
    ckpt_path_6 = args.root_dir / "best_6_hidden_layers.pt"
    if ckpt_path_6.is_file():
        net_6.load_state_dict(torch.load(ckpt_path_6))

    random_agent = Agent.random()
    print("AGS-3:", elo(Agent.from_net(net_3), random_agent))
    print("AGS-6:", elo(Agent.from_net(net_6), random_agent))
