"""
reward.py

Define the reward functions.
"""

import numpy as np

from go_stop.models.game import Game
from go_stop.train.args import args


def reward(point: int) -> float:
    """Define a reward given the actual score got from the game."""
    return (
        args.eval_win_weight * 40 * np.sign(point) + (1 - args.eval_win_weight) * point
    )


def reward_wrt_player(game: Game, player: int, reward_func=reward):
    """Return the reward w.r.t. given player and given reward function."""

    if game.state.winner is None:
        return reward_func(0)

    point = game.state.scores[game.state.winner]
    if player != game.state.winner:
        point *= -1

    return reward_func(point)
