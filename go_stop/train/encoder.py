"""
encoder.py

Encodes a game into a tensor.
"""

from typing import List, Optional, TypeVar

import torch
from torch import Tensor

from ..constants.card import go_stop_cards
from ..models.card import Card, SpecialCard
from ..models.card_list import CardList
from ..models.game import Game
from ..models.player import Player, get_opponent
from ..utils.list import flatten

Type = TypeVar("Type")

"""add `bomb card` and `hidden card` to go_stop_cards"""
extended_go_stop_cards = CardList(
    go_stop_cards + [SpecialCard("bomb")] + [SpecialCard("hidden")]
)
LEN_EXTENDED_CARDS = len(extended_go_stop_cards)


def encode_card(card: Optional[Card]) -> Tensor:
    """One-hot encoding of a card"""

    # if card is None, return (0, ..., 0)
    if card is None:
        return torch.zeros(LEN_EXTENDED_CARDS)

    # make every bomb indistinguishable
    # by regarding it as having index 0
    if card.kind == "bomb":
        return torch.eye(LEN_EXTENDED_CARDS)[
            extended_go_stop_cards.index(SpecialCard("bomb"))
        ]

    # for other cards, return the corresponding one-hot vector.
    return torch.eye(LEN_EXTENDED_CARDS)[extended_go_stop_cards.index(card)]


def cat_and_sum(lst: List[Tensor], default: Tensor) -> Tensor:
    """Cat and sum the list `list` of tensors"""
    if lst == []:
        return default

    try:
        return torch.cat([item.unsqueeze(0) for item in lst]).sum(axis=0)

    except RuntimeError:
        return default


def encode_card_list(card_list: CardList) -> Tensor:
    """Sum of one-hot encodings of cards in `card_list`"""

    return cat_and_sum(
        [encode_card(card) for card in card_list],
        torch.zeros(LEN_EXTENDED_CARDS),
    )


def pad_to_length(lst: List[Type], length: int, padder: Type) -> List[Type]:
    """Pad the list to the length `length` by `padder`"""
    if len(lst) >= length:
        return lst[0:length]

    return lst + [padder] * (length - len(lst))


def hide_opponent_hand(game: Game, opponent: Player) -> List[Card]:
    """Hide invisible opponent cards"""
    len_opponent_hand = len(game.board.hands[opponent])
    visible_cards = [
        card
        for card in game.board.hands[opponent]
        if card.kind == "bomb"
        or card in flatten(game.state.shaking_histories[opponent])
    ]
    return pad_to_length(
        visible_cards, len_opponent_hand, SpecialCard("hidden")
    )


def encode_game(game: Game, player: Player) -> Tensor:
    """
    Game encoder

    Encode the game state into a tensor of shape (307,).
    To see which information is encoded, see the return statement of this method.
    """

    opponent = get_opponent(player)

    return torch.cat(
        (
            # current player
            Tensor([game.state.player]),
            # my hand
            encode_card_list(game.board.hands[player]),
            # opponent's hand
            encode_card_list(hide_opponent_hand(game, opponent)),
            # my capture field
            encode_card_list(game.board.capture_fields[player]),
            # opponent's capture field
            encode_card_list(game.board.capture_fields[opponent]),
            # center field
            encode_card_list(flatten(game.board.center_field.values())),
            # my go history
            Tensor(
                pad_to_length(
                    game.state.go_histories[player], length=9, padder=0
                ),
            ),
            # opponent's go history
            Tensor(
                pad_to_length(
                    game.state.go_histories[opponent], length=9, padder=0
                )
            ),
            # the number of my shakings
            Tensor([len(game.state.shaking_histories[player])]),
            # the number of opponent's shakings
            Tensor([len(game.state.shaking_histories[opponent])]),
            # my stacking history; e.g., [1, 3] -> [1, 0, 1, 0, ...]
            cat_and_sum(
                torch.eye(12)[
                    [m - 1 for m in game.state.stacking_histories[player]]
                ],
                default=torch.zeros(12),
            ),
            # opponent's stacking history
            cat_and_sum(
                torch.eye(12)[
                    [m - 1 for m in game.state.stacking_histories[opponent]]
                ],
                default=torch.zeros(12),
            ),
            # my score
            Tensor([game.state.scores[player]]),
            # opponent's score
            Tensor([game.state.scores[opponent]]),
        )
    )


DIM_ENCODED_GAME = encode_game(Game(), 0).size(0)
