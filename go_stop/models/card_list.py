"""
card_list.py

Implement the class for a list of cards with a bunch of methods.
"""


from __future__ import annotations
from typing import Callable, List
from .card import Card


class CardList(list):
    """List of cards."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return CardList(list.__getitem__(self, key))

    def __str__(self):
        return "[{}]".format(", ".join([card.__str__() for card in self]))

    def apply_filter(self, predicate: Callable[[Card], bool]):
        """Apply the filter `predicate`."""
        return CardList(card for card in self if predicate(card))

    def of_month(self, month: int):
        """Return the cards with the specified month only."""
        return self.apply_filter(lambda card: card.month == month)

    def except_month(self, month: int):
        """Return the cards without the specified month only."""
        return self.apply_filter(lambda card: card.month != month)

    def sorted(self) -> CardList:
        """Return the sorted `self`."""
        self.sort()
        return self

    def serialize(self) -> List[str]:
        """Serialize a CardList."""
        return [card.serialize() for card in self]

    @staticmethod
    def deserialize(data: List[str]) -> CardList:
        """Deserialize a CardList."""
        return CardList(Card.deserialize(card) for card in data)
