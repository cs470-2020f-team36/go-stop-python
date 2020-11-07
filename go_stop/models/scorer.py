from functools import reduce
from .state import State


class Scorer:
    @staticmethod
    def calculate(state: State, player: int):
        return reduce(
            lambda a, f: f(a),
            [0] + [factor.score for factor in state.score_factors[player]],
        )
