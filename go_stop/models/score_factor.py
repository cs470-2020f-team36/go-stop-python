"""
score_factor.py

Implement `ScoreFactor`.
"""


from typing import Any, Callable, Literal, Set

from .setting import Setting


Kind = Literal[
    "bright",
    "animal",
    "ribbon",
    "junk",
    "five birds",
    "blue ribbons",
    "red ribbons",
    "plant ribbons",
    "four of a month",
    "three stackings",
    "go",
    "shaking",
    "bright penalty",
    "animal penalty",
    "junk penalty",
    "go penalty",
]


class ScoreFactor(Setting):
    """Abstracts the factors which affect to the scoring."""

    kinds: Set[Kind] = {
        "bright",
        "animal",
        "ribbon",
        "junk",
        "five birds",
        "blue ribbons",
        "red ribbons",
        "plant ribbons",
        "four of a month",
        "three stackings",
        "go",
        "shaking",
        "bright penalty",
        "animal penalty",
        "junk penalty",
        "go penalty",
    }

    def __init__(
        self,
        kind: Kind,
        arg: Any = None,
    ):
        super().__init__()

        assert kind in ScoreFactor.kinds
        self.kind: Kind = kind
        self.arg: Any = arg

    @property
    def score(self) -> Callable[[int], int]:
        """Return the function representing the effect of the score factor to the score."""

        if self.kind == "bright":
            return lambda s: s + self.arg

        if self.kind in {"animal", "ribbon"}:
            return lambda s: s + max(self.arg - 4, 0)

        if self.kind == "junk":
            return lambda s: s + max(self.arg - 9, 0)

        if self.kind == "five birds":
            return lambda s: s + 5

        if self.kind in {"blue ribbons", "red ribbons", "plant ribbons"}:
            return lambda s: s + 3

        if self.kind in "four of a month":
            return lambda s: self.score_of_four_of_a_month

        if self.kind in "three stackings":
            return lambda s: self.score_of_three_stackings

        if self.kind == "go":
            return lambda s: (s + self.arg) * 2 ** max(self.arg - 2, 0)

        if self.kind == "shaking":
            return lambda s: s * 2 ** self.arg

        if self.kind in {
            "bright penalty",
            "animal penalty",
            "junk penalty",
            "go penalty",
        }:
            return lambda s: s * 2

        assert False

    def serialize(self) -> dict:
        """Serialize the score factor."""
        return {
            "kind": self.kind,
            "arg": self.arg,
        }

    @staticmethod
    def deserialize(data: dict):
        """Deserialize the score factor."""
        return ScoreFactor(data["kind"], data["arg"])
