from typing_extensions import Literal

from .state import State


class Scorer:
    @staticmethod
    def calculate(state: State, player: Literal[0, 1]) -> int:
        result = 0

        for factor in state.score_factors[player]:
            result = factor.score(result)

        return result
