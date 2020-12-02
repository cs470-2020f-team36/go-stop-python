"""
sampler.py

Implement a sampler: g |-> f_i^{-1}(f_i(g)).
"""


import copy
import random
from typing import List

from ..models.game import Game
from ..models.card_list import CardList
from ..models.player import Player, get_opponent
from ..utils.list import flatten


def sample_from_observation(game: Game, player: Player, sample_size: int) -> List[Game]:
    """
    Game sampler

    Returns a sample of games that is seemingly identical to the given player.
    """

    # sample size should be positive
    assert sample_size > 0

    opponent = get_opponent(player)

    # list of cards in the hand of the opponent
    # which is visible to the player;
    # in Go-Stop, bomb cards and cards that were shaken are visible to all
    visible_opponent_hand = game.board.hands[opponent].apply_filter(
        lambda card: (
            card.kind == "bomb"
            or card in flatten(game.state.shaking_histories[opponent])
        )
    )

    # list of cards in the hand of the opponent
    # which is hidden to the player
    hidden_opponent_hand = game.board.hands[opponent].apply_filter(
        lambda card: card not in visible_opponent_hand
    )

    # list of all hidden cards
    hidden_cards = CardList(hidden_opponent_hand + game.board.drawing_pile)

    # resulting list of sampled games
    res = []

    for _ in range(sample_size):
        # shuffle hidden cards and redistribute them
        random.shuffle(hidden_cards)

        # deepcopy the game
        new_game = copy.deepcopy(game)

        # replace the hidden part of the opponent's hand
        new_game.board.hands[opponent] = CardList(
            visible_opponent_hand + hidden_cards[: len(hidden_opponent_hand)]
        )

        # replace the drawing pile by remaining cards
        new_game.board.drawing_pile = hidden_cards[len(hidden_opponent_hand) :]

        # sort the board
        new_game.sort_board()

        res.append(new_game)

    return res
