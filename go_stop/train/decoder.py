"""
encoder.py

Encodes a game into a tensor.
"""

from typing import List, Optional, TypeVar

import torch
from torch import Tensor

from .encoder import extended_go_stop_cards, LEN_EXTENDED_CARDS
from ..constants.card import go_stop_cards
from ..models.card import Card, SpecialCard
from ..models.card_list import CardList
from ..models.game import Game
from ..models.player import Player, get_opponent
from ..utils.list import flatten

Type = TypeVar("Type")


def decode_card_list(tensor: Tensor) -> CardList:
    """(Almost) Inverse of `encode_card_list`."""

    return flatten(
        [[extended_go_stop_cards[i]] * int(n) for i, n in enumerate(tensor.tolist())]
    )


def pad_to_length(lst: List[Type], length: int, padder: Type) -> List[Type]:
    """Pad the list to the length `length` by `padder`"""
    if len(lst) >= length:
        return lst[0:length]

    return lst + [padder] * (length - len(lst))


def decode_game(tensor: Tensor) -> Game:
    """
    Game decoder

    Decode the game state from a tensor of shape (307,),
    noting that it is not a full inverse of `encode_game`.
    """

    my_hand = decode_card_list(tensor[:LEN_EXTENDED_CARDS])
    opp_hand = decode_card_list(tensor[LEN_EXTENDED_CARDS : 2 * LEN_EXTENDED_CARDS])
    my_cap_field = decode_card_list(
        tensor[2 * LEN_EXTENDED_CARDS : 3 * LEN_EXTENDED_CARDS]
    )
    opp_cap_field = decode_card_list(
        tensor[3 * LEN_EXTENDED_CARDS : 4 * LEN_EXTENDED_CARDS]
    )
    center_field = decode_card_list(
        tensor[4 * LEN_EXTENDED_CARDS : 5 * LEN_EXTENDED_CARDS]
    )
    my_go_history = tensor[5 * LEN_EXTENDED_CARDS : 5 * LEN_EXTENDED_CARDS + 9]
    opp_go_history = tensor[5 * LEN_EXTENDED_CARDS + 9 : 5 * LEN_EXTENDED_CARDS + 18]
    my_num_shaking = tensor[5 * LEN_EXTENDED_CARDS + 18]
    opp_num_shaking = tensor[5 * LEN_EXTENDED_CARDS + 19]
    my_stacking_history = tensor[
        5 * LEN_EXTENDED_CARDS + 20 : 5 * LEN_EXTENDED_CARDS + 32
    ]
    opp_stacking_history = tensor[
        5 * LEN_EXTENDED_CARDS + 32 : 5 * LEN_EXTENDED_CARDS + 44
    ]
    my_score = tensor[5 * LEN_EXTENDED_CARDS + 44]
    opp_score = tensor[5 * LEN_EXTENDED_CARDS + 45]

    print(
        "my_hand             ", my_hand,
    )
    print(
        "opp_hand            ", opp_hand,
    )
    print(
        "my_cap_field        ", my_cap_field,
    )
    print(
        "opp_cap_field       ", opp_cap_field,
    )
    print(
        "center_field        ", center_field,
    )
    print(
        "my_go_history       ", my_go_history,
    )
    print(
        "opp_go_history      ", opp_go_history,
    )
    print(
        "my_num_shaking      ", my_num_shaking,
    )
    print(
        "opp_num_shaking     ", opp_num_shaking,
    )
    print(
        "my_stacking_history ", my_stacking_history,
    )
    print(
        "opp_stacking_history", opp_stacking_history,
    )
    print(
        my_score,
    )
    print(
        opp_score,
    )
