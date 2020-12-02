from typing import Literal

from .player import Player
from .state import State


class Scorer:
    @staticmethod
    def calculate(state: State, player: Player) -> int:
        result = 0

        for factor in state.score_factors[player]:
            result = factor.score(result)

        return result
