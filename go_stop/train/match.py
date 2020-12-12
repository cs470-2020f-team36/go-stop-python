"""
train.py

Train the neural network.
Note that this project does not use GPU,
as the most time-consuming step in this project is MCTS,
which is difficult to accelerate using GPU.
"""

from typing import List, Optional

import copy
import numpy as np
import pickle
import torch
import tqdm

from go_stop.models.agent import Agent
from go_stop.models.game import Game
from go_stop.train.args import args
from go_stop.train.reward import reward_wrt_player


def match_agents(
    agent_a: Agent,
    agent_b: Agent,
    num_evaluation_games: int = args.num_evaluation_games,
    games: Optional[List[Game]] = None,
) -> List[float]:
    """Match two agents each other by playing with games in `games`."""

    if games is None:
        try:
            with open(args.root_dir / "games.pickle", "rb") as games_pickle:
                games = pickle.load(games_pickle)[:num_evaluation_games // 2]
        except:
            print("No games predefined.")
            return []

    with torch.no_grad():
        # The array of scores for the agent A.
        scores = []
        is_first_doing_better = []
        agents = [agent_a, agent_b]

        # play `num_random_games` games
        for game in tqdm.tqdm(games, desc="match_agents", leave=False):
            copied_game = copy.deepcopy(game)

            # play with `game`
            while not game.state.ended:
                action = agents[game.state.player].query(game)
                game.play(action)

            scores.append(reward_wrt_player(game, 0, lambda p: p))

            # play with `copied_game` with swapping players
            while not copied_game.state.ended:
                action = agents[1 - copied_game.state.player].query(copied_game)
                copied_game.play(action)

            scores.append(reward_wrt_player(copied_game, 1, lambda p: p))

            is_first_doing_better.append(np.sign(scores[-2] + scores[-1]))

        print(
            "The first one is better on",
            len([a for a in is_first_doing_better if a > 0]),
            "games",
        )
        print(
            "The second one is better on",
            len([a for a in is_first_doing_better if a < 0]),
            "games",
        )
        return scores


def is_first_better(points: List[float]) -> bool:
    """Get the list of scores and return whether the first agent is better."""

    sample_mean = np.mean(points)

    if sample_mean > 0:
        return True

    return False
