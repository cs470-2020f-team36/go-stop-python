from functools import reduce
from typing import Any, List, cast

from .state import State


class Scorer:
    @staticmethod
    def calculate(state: State, player: int):
        result = 0

        for factor in state.score_factors[player]:
            result = factor.score(result)

        return result
