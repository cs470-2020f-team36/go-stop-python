"""
board.py

Implement the board of Go-Stop, which is the abstraction of all card fields.
"""

from copy import copy
import random
from typing import Dict, List, Set, Tuple

from ..constants.card import go_stop_cards
from .card import Card, BonusCard
from .card_list import CardList
from .setting import Setting


class Board(Setting):
    """
    Abstraction of all card fields.
    """

    def __init__(self, starting_player: int = 0):
        super().__init__()

        while True:
            # shuffle the drawing pile
            self.drawing_pile: CardList = copy(go_stop_cards)
            random.shuffle(self.drawing_pile)

            # hands
            self.hands: List[CardList] = [
                self.drawing_pile[0:10],
                self.drawing_pile[10:20],
            ]

            # temporary center field
            temp_center_field: CardList = self.drawing_pile[20:28]

            # remaining drawing pile
            self.drawing_pile = self.drawing_pile[28:]

            # capture fields
            self.capture_fields: List[CardList] = [CardList(), CardList()]

            temp_center_field = self._move_cards_at_beginning(
                starting_player, temp_center_field
            )

            if not self._is_reset_necessary(temp_center_field):
                break

        # sort card lists
        temp_center_field.sort()
        self.center_field: Dict[int, CardList] = dict(
            (month, temp_center_field.of_month(month)) for month in range(1, 13)
        )

        for month in range(1, 13):
            self.center_field[month].sort()

        self.sort()

    def _is_reset_necessary(self, temp_center_field: CardList) -> bool:
        """
        if all players have a four-of-a-month
        or the center field has a four-of-a-month
        (only if proceed_when_center_field_has_a_four_of_a_month is not set),
        reset the board.
        """

        four_of_a_month = self.four_of_a_month()

        if self.proceed_when_center_field_has_a_four_of_a_month:
            return (
                len(
                    [
                        player
                        for player in [0, 1]
                        if len(four_of_a_month[player]) > 0
                    ]
                )
                == 2
            )

        return len(
            [player for player in [0, 1] if len(four_of_a_month[player]) > 0]
        ) == 2 or any(
            len(temp_center_field.of_month(month)) == 4
            for month in range(1, 13)
        )

    def _move_cards_at_beginning(
        self, starting_player: int, temp_center_field: CardList
    ) -> CardList:
        """
        A private method which moves bonus cards and four-of-a-months
        at the beginning to the starting_player.

        In detail,

        1. If the center field has bonus cards,
           place them into starting_player's capture field and
           turn the top card of the drawing pile into the center field.

        2. If the center field has four of a month,
           place them into starting_player's capture field and
           turn the top card of the drawing pile into the center field.
        [2: only when self.proceed_when_center_field_has_a_four_of_a_month is True]

        3. Repeat 1 [or 2] until there are no changes anymore.
        """

        changed = False

        for bonus_multiple in {2, 3}:
            if BonusCard(bonus_multiple) in temp_center_field:
                temp_center_field.remove(BonusCard(bonus_multiple))
                card = self.drawing_pile.pop(0)
                temp_center_field.append(card)
                self.capture_fields[starting_player].append(
                    BonusCard(bonus_multiple)
                )
                changed = True

        if self.proceed_when_center_field_has_a_four_of_a_month:
            for month in range(1, 13):
                if len(temp_center_field.of_month(month)) == 4:
                    temp_center_field = temp_center_field.except_month(month)
                    cards = self.drawing_pile[0:4]
                    temp_center_field.extend(cards)
                    self.drawing_pile = self.drawing_pile[4:]
                    self.capture_fields[starting_player].extend(
                        go_stop_cards.of_month(month)
                    )
                    changed = True

        if changed:
            return self._move_cards_at_beginning(
                starting_player, temp_center_field
            )

        return temp_center_field

    def four_of_a_month(self) -> Tuple[Set[int], Set[int]]:
        """
        Returns information about which four-of-a-months are attained by players.
        """

        four_of_a_month: Tuple[Set[int], Set[int]] = (set(), set())

        for player in [0, 1]:
            for month in range(1, 13):
                if set(go_stop_cards.of_month(month)).issubset(
                    self.hands[player]
                ):
                    four_of_a_month[player].add(month)

        return four_of_a_month

    def sort(self) -> None:
        """Sort hands of players."""

        for player in [0, 1]:
            self.hands[player].sort()

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

        center_field = data["center_field"]
        board.center_field = dict(
            (
                month,
                CardList(
                    Card.deserialize(s) for s in center_field[month.__str__()]
                )
                if month.__str__() in center_field
                else CardList([]),
            )
            for month in range(1, 13)
        )

        board.drawing_pile = CardList(
            Card.deserialize(s) for s in data["drawing_pile"]
        )

        for player in [0, 1]:
            board.hands[player].sort()

        return board
