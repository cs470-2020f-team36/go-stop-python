"""
player.py

Define the player type `Player` and the method `get_opponent`.
"""

from typing import Literal, cast


Player = Literal[0, 1]


def get_opponent(player: Player) -> Player:
    """Return the opponent of `player`, which is 1 - player."""
    return cast(Player, 1 - player)
