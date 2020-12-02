"""
action.py

Implement an action during a Go-Stop game.
"""


from abc import ABC, abstractmethod
from typing import Any, Literal

from .card import Card
from ..constants.card import go_stop_cards


Kind = Literal[
    "throw",
    "throw bomb",
    "bomb",
    "shakable",
    "shaking",
    "select match",
    "four of a month",
    "go",
    "move animal 9",
]


class Action(ABC):
    """
    Actions

    There are maximum 167 (50 + 1 + 12 + 48 + 2 + 48 + 2 + 2 + 2) possible actions.
    """

    def __init__(self, kind: Kind, arg: Any):
        self.kind = kind
        self.arg = arg

    def __eq__(self, obj: object):
        return (
            isinstance(obj, Action)
            and obj.kind == self.kind
            and obj.arg == self.arg
        )

    def __str__(self):
        return (
            f"{self.kind} {str(self.arg)}"
            if self.arg is not None
            else self.kind
        )

    @abstractmethod
    def serialize(self) -> dict:
        """Serialize an action"""

    @staticmethod
    def deserialize(data: dict):
        """Deserialize an action"""

        if data["kind"] == "throw":
            return ActionThrow(Card.deserialize(data["card"]))

        if data["kind"] == "throw bomb":
            return ActionThrowBomb()

        if data["kind"] == "bomb":
            return ActionBomb(data["month"])

        if data["kind"] == "shakable":
            return ActionShakable(Card.deserialize(data["card"]))

        if data["kind"] == "shaking":
            return ActionShaking(data["option"])

        if data["kind"] == "select match":
            return ActionSelectMatch(Card.deserialize(data["match"]))

        if data["kind"] == "four of a month":
            return ActionFourOfAMonth(data["option"])

        if data["kind"] == "move animal 9":
            return ActionMoveAnimal9(data["option"])

        if data["kind"] == "go":
            return ActionGo(data["option"])

        assert False


class ActionThrow(Action):
    """
    ("throw", card) -> maximum 50 actions (50 cards)
    """

    def __init__(self, card: Card):
        super().__init__("throw", card)
        self.card = card

    def serialize(self) -> dict:
        return {"kind": self.kind, "card": self.card.serialize()}


class ActionThrowBomb(Action):
    """
    ("throw bomb") -> maximum 1 action
    """

    def __init__(self):
        super().__init__("throw bomb", None)

    def serialize(self) -> dict:
        return {"kind": self.kind}


class ActionBomb(Action):
    """
    ("bomb", month) -> maximum 12 actions (12 months)
    """

    def __init__(self, month: int):
        super().__init__("bomb", month)
        self.month = month

    def serialize(self) -> dict:
        return {"kind": self.kind, "month": self.month}


class ActionShakable(Action):
    """
    ("shakable", card) -> maximum 48 actions (48 cards with month)
    """

    def __init__(self, card: Card):
        super().__init__("shakable", card)
        self.card = card

    def serialize(self) -> dict:
        return {"kind": self.kind, "card": self.card.serialize()}


class ActionShaking(Action):
    """
    ("shaking", option) -> maximum 2 actions (bool)
    """

    def __init__(self, option: bool):
        super().__init__("shaking", option)
        self.option = option

    def serialize(self) -> dict:
        return {"kind": self.kind, "option": self.option}


class ActionSelectMatch(Action):
    """
    ("shaking", match) -> maximum 48 actions (48 cards with month)
    """

    def __init__(self, match: Card):
        super().__init__("select match", match)
        self.match = match

    def serialize(self) -> dict:
        return {"kind": self.kind, "match": self.match.serialize()}


class ActionFourOfAMonth(Action):
    """
    Go-Stop action for deciding whether the player claim a four-of-a-month or not

    ("four of a month", option) -> maximum 2 actions
    """

    def __init__(self, option: bool):
        super().__init__("four of a month", option)
        self.option = option

    def serialize(self) -> dict:
        return {"kind": self.kind, "option": self.option}


class ActionMoveAnimal9(Action):
    """
    ("move animal 9", option) -> maximum 2 actions (bool)
    """

    def __init__(self, option: bool):
        super().__init__("move animal 9", option)
        self.option = option

    def serialize(self) -> dict:
        return {"kind": self.kind, "option": self.option}


class ActionGo(Action):
    """
    ("go", option) -> maximum 2 actions (bool)
    """

    def __init__(self, option: bool):
        super().__init__("go", option)
        self.option = option

    def serialize(self) -> dict:
        return {"kind": self.kind, "option": self.option}


all_actions = [
    *[ActionThrow(card) for card in go_stop_cards],
    ActionThrowBomb(),
    *[ActionBomb(month) for month in range(1, 13)],
    *[ActionShakable(card) for card in go_stop_cards if card.month is not None],
    *[ActionShaking(option) for option in {True, False}],
    *[
        ActionSelectMatch(match)
        for match in go_stop_cards
        if match.month is not None
    ],
    *[ActionFourOfAMonth(option) for option in {True, False}],
    *[ActionMoveAnimal9(option) for option in {True, False}],
    *[ActionGo(option) for option in {True, False}],
]


def get_action_index(action: Action) -> int:
    """Return the index of an action"""

    return all_actions.index(action)


NUM_ACTIONS = len(all_actions)
