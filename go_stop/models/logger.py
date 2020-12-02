"""
logger.py

Implement a logger of a game.
"""


from typing import Any, Literal


Kind = Literal[
    "turn change",  # arg: (int, int)
    "throw",  # arg: Card
    "flip",  # arg: Card
    "bomb",  # arg: int
    "shaking",  # arg: CardList
    "append to hand",  # arg: (int, CardList)
    "append to center field",  # arg: CardList
    "append to capture field",  # arg: (int, CardList)
    "take junk from opponent",  # arg: Card
    "single match",  # arg: (Card, Card)
    "double matches",  # arg: (Card, CardList)
    "triple matches",  # arg: (Card, CardList)
    "select match",  # arg: (Card, Card)
    "discard and match",  # arg: (Card, Card)
    "stacking",  # arg: CardList
    "ttadak",  # arg: int
    "move animal 9",  # arg: None
    "clear",  # arg: None
    "go",  # arg: bool
    "four of a month",  # arg: CardList
]


class LoggerItem:
    """An item of a logger"""

    def __init__(self, kind: Kind, arg):
        self.kind = kind
        self.arg = arg

    def __str__(self):
        if self.kind == "turn change":
            return "TURN CHANGE: P{} -> P{}".format(self.arg[0], self.arg[1])
        if self.kind == "throw":
            return "THROW: {}".format(str(self.arg))
        if self.kind == "flip":
            return "FLIP: {}".format(str(self.arg))
        if self.kind == "bomb":
            return "BOMB: {}".format(self.arg)
        if self.kind == "shaking":
            return "SHAKING: {}".format(self.arg)
        if self.kind == "append to hand":
            return "APPEND TO P{}'s HAND: {}".format(
                self.arg[0], str(self.arg[1])
            )
        if self.kind == "append to center field":
            return "APPEND TO CENTER FIELD: {}".format(str(self.arg))
        if self.kind == "append to capture field":
            return "APPEND TO P{}'s CAPTURE FIELD: {}".format(
                self.arg[0], str(self.arg[1])
            )
        if self.kind == "take junk from opponent":
            return "TAKE JUNK: {}".format(self.arg)
        if self.kind == "single match":
            return "SINGLE MATCH: {} with {}".format(self.arg[0], self.arg[1])
        if self.kind == "double matches":
            return "DOUBLE MATCHES: {} with {}".format(self.arg[0], self.arg[1])
        if self.kind == "triple matches":
            return "TRIPLE MATCHES: {} with {}".format(self.arg[0], self.arg[1])
        if self.kind == "select match":
            return "SELECT MATCH: {} with {}".format(self.arg[0], self.arg[1])
        if self.kind == "discard and match":
            return "DISCARD-AND-MATCH: {} with {}".format(
                self.arg[0], self.arg[1]
            )
        if self.kind == "stacking":
            return "STACKING: {}".format(self.arg)
        if self.kind == "ttadak":
            return "TTADAK: {}".format(self.arg)
        if self.kind == "move animal 9":
            return "MOVE ANIMAL 9: {}".format(self.arg)
        if self.kind == "clear":
            return "CLEAR"
        if self.kind == "go":
            return "GO" if self.arg else "STOP"
        if self.kind == "four of a month":
            return "FOUR OF A MONTH: {}".format(str(self.arg))


class Logger:
    """Logger class"""

    def __init__(self):
        self.logs = []

    def __str__(self):
        return "[{}]".format(", ".join([str(item) for item in self.logs]))

    def log(self, kind: Kind, arg: Any = None):
        """Append a log item to self.logs"""

        self.logs.append(LoggerItem(kind, arg))

    def serialize(self):
        """Serialize the logger"""
        return [str(item) for item in self.logs]
