from typing import Any, Callable
from .setting import Setting


class ScoreFactor(Setting):
    kinds = {
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

    def __init__(self, kind: str, arg: Any = None):
        super().__init__()

        assert kind in ScoreFactor.kinds
        self.kind = kind
        self.arg = arg

    @property
    def score(self) -> Callable[[int], int]:
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

    def serialize(self) -> dict:
        return {
            "kind": self.kind,
            "arg": self.arg,
        }

    @staticmethod
    def deserialize(data: dict):
        return ScoreFactor(data["kind"], data["arg"])