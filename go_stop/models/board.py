from typing import List, Set
import json
from copy import copy, deepcopy
import random
from ..constants.card import go_stop_cards
from .card import Card, BrightCard, AnimalCard, RibbonCard, JunkCard, BonusCard
from .card_list import CardList
from .setting import Setting

class Board(Setting):
    """
    Abstracting all card fields
    """

    def __init__(self):
        # shuffle the drawing pile
        self.drawing_pile = copy(go_stop_cards)
        random.shuffle(self.drawing_pile)

        # hands
        self.hands = [
            self.drawing_pile[0:10],
            self.drawing_pile[10:20],
        ]

        # center field
        center_field = self.drawing_pile[20:28]

        # drawing pile
        self.drawing_pile = self.drawing_pile[28:]

        # capture fields
        self.capture_fields = [CardList(), CardList()]

        center_field = self._move_cards_at_beginning(center_field)

        # sort card lists
        center_field.sort()
        self.center_field = dict(
            (month, center_field.of_month(month)) for month in range(1, 13)
        )

        for month in range(1, 13):
            self.center_field[month].sort()

        self.sort()

    def _move_cards_at_beginning(self, center_field: CardList) -> CardList:
        """
        A private method which moves bonus cards and four-of-a-months
        at the beginning to the first player.

        In detail,

        1. If the center field has bonus cards,
           place them into player 0's capture field and
           turn the top card of the drawing pile into the center field.

        2. If the center field has four of a month,
           place them into player 0's capture field and
           turn the top card of the drawing pile into the center field.
        [2: only when self.proceed_when_center_field_has_a_four_of_a_month is True]

        3. Repeat 1 [or 2] until there are no changes anymore.
        """

        changed = False

        for n in {2, 3}:
            if BonusCard(n) in center_field:
                center_field.remove(BonusCard(n))
                card = self.drawing_pile.pop(0)
                center_field.append(card)
                self.capture_fields[0].append(BonusCard(n))
                changed = True

        if self.proceed_when_center_field_has_a_four_of_a_month:
            for month in range(1, 13):
                if len(center_field.of_month(month)) == 4:
                    center_field = center_field.except_month(month)
                    cards = self.drawing_pile[0:4]
                    center_field.extend(cards)
                    self.drawing_pile = self.drawing_pile[4:]
                    self.capture_fields[0].extend(go_stop_cards.of_month(month))
                    changed = True

        if changed:
            return self._move_cards_at_beginning(center_field)

        return center_field

    def four_of_a_month(self) -> List[Set[int]]:
        """
        Returns information about which four-of-a-months are attained by players.
        """

        four_of_a_month = [set(), set()]

        for player in {0, 1}:
            for month in range(1, 13):
                if set(go_stop_cards.of_month(month)).issubset(
                    self.hands[player]
                ):
                    four_of_a_month[player].add(month)

        return four_of_a_month

    def sort(self) -> None:
        for player in {0, 1}:
            self.hands[player].sort()

    def render(self, mode: str) -> None:
        """
        Render the board.
        """

        if mode == "text":
            for player in {0, 1}:
                print("P{}의 패:".format(player), self.hands[player])
                print("P{}이 얻은 패:".format(player), self.capture_fields[player])
            print("깔린 패:")
            for month in range(1, 13):
                print(
                    "{}: {}".format(month, CardList(self.center_field[month]))
                )
            print("패 더미:", self.drawing_pile)

    def serialize(self) -> dict:
        """
        Serialize the board.
        """

        return {
            "hands": [
                self.hands[0].serialize(),
                self.hands[1].serialize(),
            ],
            "capture_fields": [
                self.capture_fields[0].serialize(),
                self.capture_fields[1].serialize(),
            ],
            "center_field": dict(
                (str(month), self.center_field[month].serialize())
                for month in range(1, 13)
                if self.center_field[month] != []
            ),
            "drawing_pile": self.drawing_pile.serialize(),
        }

    @staticmethod
    def deserialize(data: dict):
        """
        Deserialize the board.
        """

        board = Board()

        board.hands = [
            CardList(Card.deserialize(s) for s in data["hands"][0]),
            CardList(Card.deserialize(s) for s in data["hands"][1]),
        ]

        board.capture_fields = [
            CardList(Card.deserialize(s) for s in data["capture_fields"][0]),
            CardList(Card.deserialize(s) for s in data["capture_fields"][1]),
        ]

        d = data["center_field"]
        board.center_field = dict(
            (
                month,
                CardList(
                    Card.deserialize(s)
                    for s in (
                        d[month.__str__()] if month.__str__() in d else []
                    )
                ),
            )
            for month in range(1, 13)
        )

        board.drawing_pile = CardList(
            Card.deserialize(s) for s in data["drawing_pile"]
        )

        for player in {0, 1}:
            board.hands[player].sort()

        return board