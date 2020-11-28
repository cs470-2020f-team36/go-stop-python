from abc import ABC, abstractmethod
from typing import Any, Optional
from typing_extensions import Literal


class Card(ABC):
    """
    ABC for cards.
    """

    def __init__(self, kind: str, month: Optional[int], index: int):
        self.kind = kind
        self.month = month
        self.index = index

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, obj: Any):
        return (
            isinstance(obj, Card)
            and obj.kind == self.kind
            and obj.month == self.month
            and obj.index == self.index
        )

    def __hash__(self):
        return hash((self.kind, self.month, self.index))

    def _to_level(self):
        """
        A private method used to sort cards.
        """

        kind_to_level = {
            "bright": 1,
            "animal": 2,
            "ribbon": 3,
            "junk": 4,
            "bonus": 5,
            "bomb": 6,
        }
        return (
            (13 if self.month == None else self.month) * 1000
            + kind_to_level[self.kind] * 100
            + self.index
        )

    def __lt__(self, obj):
        assert isinstance(obj, Card)
        return self._to_level() < obj._to_level()

    def serialize(self: "Card") -> str:
        """
        Serialize a card.
        """

        if self.kind == "bright":
            assert self.month is not None
            return "B{:02d}".format(self.month)

        if self.kind == "animal":
            assert self.month is not None
            return "A{:02d}".format(self.month)

        if self.kind == "ribbon":
            assert self.month is not None
            return "R{:02d}".format(self.month)

        if self.kind == "junk":
            assert self.month is not None
            return "J{:02d}{}".format(self.month, self.index)

        if self.kind == "bonus":
            # pylint: disable=no-member
            return "+{}".format(self.multiple)

        if self.kind == "bomb":
            return "*{}".format(self.index)

        return "?"

    @staticmethod
    def deserialize(serial: str) -> "Card":
        """
        Deserialize to a card.
        """

        if serial[0] == "B":
            return BrightCard(int(serial[1:]))

        if serial[0] == "A":
            return AnimalCard(int(serial[1:]))

        if serial[0] == "R":
            return RibbonCard(int(serial[1:]))

        if serial[0] == "J":
            month = int(serial[1:3])
            index = 0 if month == 12 else int(serial[3])
            multiple = 2 if (month == 11 and index == 2) or month == 12 else 1
            return JunkCard(month, index, multiple)

        if serial[0] == "+":
            return BonusCard(int(serial[1:]))

        assert serial[0] == "*"
        return SpecialCard("bomb", int(serial[1:]))


class BrightCard(Card):
    months = {1, 3, 8, 11, 12}

    def __init__(self, month: int):
        assert month in BrightCard.months
        super().__init__("bright", month, 0)

    def __str__(self):
        return "{}월 광".format(self.month)


class AnimalCard(Card):
    months = {2, 4, 5, 6, 7, 8, 9, 10, 12}

    def __init__(self, month: int):
        assert month in AnimalCard.months
        super().__init__("animal", month, 0)

    def __str__(self):
        return "{}월 열끗".format(self.month)


class RibbonCard(Card):
    months = {1, 2, 3, 4, 5, 6, 7, 9, 10, 12}

    def __init__(self, month: int):
        assert month in RibbonCard.months
        super().__init__("ribbon", month, 0)

        if month in {1, 2, 3}:
            self.ribbon_color: Optional[Literal["red", "blue", "plant"]] = "red"
        elif month in {4, 5, 7}:
            self.ribbon_color = "plant"
        elif month in {6, 9, 10}:
            self.ribbon_color = "blue"
        else:
            self.ribbon_color = None

    def __str__(self):
        translation = {"blue": "청", "red": "홍", "plant": "초"}
        return "{}월 단".format(self.month) + (
            " ({}단)".format(translation[self.ribbon_color])
            if self.ribbon_color != None
            else ""
        )


class JunkCard(Card):
    def __init__(self, month: int, index: int = 0, multiple: int = 1):
        assert multiple == 1 or multiple == 2
        if month == 11 and multiple == 2:
            assert index == 2
        elif month == 12:
            assert multiple == 2
            assert index == 0
        else:
            assert multiple == 1
            assert index < 2

        self.multiple = multiple
        super().__init__("junk", month, index)

    def __str__(self):
        translation = {1: "", 2: "쌍"}
        return "{}월 {}피{}".format(
            self.month,
            translation[self.multiple],
            " ({})".format(self.index) if self.multiple == 1 else "",
        )


class BonusCard(Card):
    def __init__(self, multiple: Literal[2, 3]):
        self.multiple = multiple
        super().__init__("bonus", None, multiple - 2)

    def __str__(self):
        translation = {2: "투", 3: "쓰리"}
        return "보너스 {}피".format(translation[self.multiple])


class SpecialCard(Card):
    def __init__(self, kind: Literal["bomb", "hidden"], index: int):
        super().__init__(kind, None, index)

    def __str__(self):
        return "뒷면" if self.kind == "hidden" else "폭탄 {}".format(self.index)
