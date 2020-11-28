from typing import Callable, List
from .card import Card


class CardList(list):
    """
    List of cards.
    """

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return CardList(list.__getitem__(self, key))

    def __str__(self):
        return "[{}]".format(", ".join([card.__str__() for card in self]))

    def apply_filter(self, predicate: Callable[[Card], bool]):
        return CardList(card for card in self if predicate(card))

    def of_month(self, month: int):
        return self.apply_filter(lambda card: card.month == month)

    def except_month(self, month: int):
        return self.apply_filter(lambda card: card.month != month)

    def sorted(self) -> "CardList":
        self.sort()
        return self

    def serialize(self) -> List[str]:
        return [card.serialize() for card in self]

    @staticmethod
    def deserialize(data) -> "CardList":
        return CardList(Card.deserialize(card) for card in data)