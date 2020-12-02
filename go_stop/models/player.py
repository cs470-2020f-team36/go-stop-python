from typing import Literal, cast


Player = Literal[0, 1]


def get_opponent(player: Player) -> Player:
    return cast(Player, 1 - player)
