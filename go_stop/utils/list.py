"""
Utility functions related to lists.
"""

from typing import Iterable

from go_stop.models.card_list import CardList


def flatten(iterable: Iterable[CardList]) -> CardList:
    """Flattens an iterable of CardLists into a single CardList."""
    return CardList([el for sublist in iterable for el in sublist])
