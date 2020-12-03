"""
scorer.py

Define a function calculating the score from the list of score factors.
"""

from .player import Player
from .state import State


def calculate_score(state: State, player: Player) -> int:
    """Calculates the score from the list of score factors."""

    result = 0

    for factor in state.score_factors[player]:
        result = factor.score(result)

    return result
