"""
state.py

Gather all the states during a Go-Stop game.
"""

from typing import List, Optional, Set, Tuple, Union, cast

from .card import Card
from .card_list import CardList
from .player import Player
from .score_factor import ScoreFactor


# pylint: disable=too-many-instance-attributes


class State:
    """State class."""

    def __init__(self, starting_player: Player = 0):
        """Index of the starting player."""
        self.starting_player: Player = starting_player

        """Index of the current player."""
        self.player: Player = starting_player

        """Index of the bomb used to guarantee the uniqueness of `index`."""
        self.bomb_increment: int = 0

        """
        Histories of points at Go's of players.
        For instance, [[8, 10, 32], [9]] means the player 0 claimed Go three times,
        and the player 1 did once, where the points are calculated after applying
        the score advantage of Go's, that is,
            P0: "Go" at 7 -> became 8, "Go" at 9 -> became "10", "Go" at 15 -> became "32" and
            P1: "Go" at 8 -> became 9
        """
        self.go_histories: List[List[int]] = [[], []]

        """Save information of among which cards the player will select one."""
        self.select_match: Union[
            None,
            Tuple[
                Card,  # card thrown
                CardList,  # cards which are matched
                Optional[
                    Tuple[
                        CardList,  # list of cards which were captured before flip
                        Card,  # card flipped
                        CardList,  # captured bonus cards by flipping top card of drawing pile
                        int,  # number of junk cards taken from opponent
                    ],
                ],
            ],
        ] = None

        """Save information about which cards the player will shake with."""
        self.shaking: Optional[
            Tuple[
                Card,  # card to shake
                CardList,  # collection of cards in a shaking (length >= 3)
            ]
        ] = None

        """
        shaking_histories: List[List[CardList]]
        How many shakings are done by players.
        """
        self.shaking_histories: List[List[CardList]] = [[], []]

        """
        stacking_histories: List[Set[int]]
        Which stackings are made by players.
        """
        self.stacking_histories: List[Set[int]] = [set(), set()]

        """
        score_factors: List[List[ScoreFactor]]
        Sets of score factors, which are updated at almost end of each turn.
        ScoreFactor of kind "go" and "penalties" should be at the very last.
        """
        self.score_factors: List[List[ScoreFactor]] = [
            [],
            [],
        ]

        """
        scores: List[int]
        """
        self.scores: List[int] = [0, 0]

        """
        animal_9_moved: Optional[bool]
        Whether the animal card of September has moved to the junk field.
        `None` if it is not queried to move the animal card of September.
        """
        self.animal_9_moved: Optional[bool] = None

        """
        ended: bool
        Whether the game ended.
        """
        self.ended: bool = False

        """
        winner: Optional[Player]
        Who won the game, if it is ended.
        """
        self.winner: Optional[Player] = None

    def serialize(self) -> dict:
        """
        Serialize the state.
        """

        return {
            "starting_player": self.starting_player,
            "player": self.player,
            "bomb_increment": self.bomb_increment,
            "go_histories": self.go_histories,
            "select_match": None
            if self.select_match is None
            else (  # before flip
                None
                if self.select_match[0] is None
                else cast(Card, self.select_match[0]).serialize(),
                cast(CardList, self.select_match[1]).serialize(),
                None,
            )
            if self.select_match[2] is None
            else (
                None
                if self.select_match[0] is None
                else cast(Card, self.select_match[0]).serialize(),
                cast(CardList, self.select_match[1]).serialize(),
                (
                    self.select_match[2][0].serialize(),
                    self.select_match[2][1].serialize(),
                    self.select_match[2][2].serialize(),
                    self.select_match[2][3],
                ),
            ),
            "shaking": None
            if self.shaking is None
            else [
                self.shaking[0].serialize(),
                self.shaking[1].serialize(),
            ],
            "shaking_histories": [
                [l.serialize() for l in self.shaking_histories[0]],
                [l.serialize() for l in self.shaking_histories[1]],
            ],
            "stacking_histories": [
                list(self.stacking_histories[0]),
                list(self.stacking_histories[1]),
            ],
            "score_factors": [
                [f.serialize() for f in self.score_factors[0]],
                [f.serialize() for f in self.score_factors[1]],
            ],
            "scores": self.scores,
            "animal_9_moved": self.animal_9_moved,
            "ended": self.ended,
            "winner": self.winner,
        }

    @staticmethod
    def deserialize(data: dict):
        """
        Deserialize the state.
        """

        state = State()

        state.starting_player = data["starting_player"]
        state.player = data["player"]
        state.bomb_increment = data["bomb_increment"]
        state.go_histories = [
            cast(List[int], data["go_histories"][0]),
            cast(List[int], data["go_histories"][1]),
        ]
        state.select_match = (
            None
            if data["select_match"] is None
            else (
                Card.deserialize(data["select_match"][0]),
                CardList.deserialize(data["select_match"][1]),
                None
                if data["select_match"][2] is None
                else (
                    CardList.deserialize(data["select_match"][2][0]),
                    Card.deserialize(data["select_match"][2][1]),
                    CardList.deserialize(data["select_match"][2][2]),
                    cast(int, data["select_match"][2][3]),
                ),
            )
        )
        state.shaking = (
            None
            if data["shaking"] is None
            else (
                Card.deserialize(data["shaking"][0]),
                CardList.deserialize(data["shaking"][1]),
            )
        )
        state.shaking_histories = [
            CardList.deserialize(data["stacking_histories"][0]),
            CardList.deserialize(data["stacking_histories"][1]),
        ]
        state.stacking_histories = [
            set(data["stacking_histories"][0]),
            set(data["stacking_histories"][1]),
        ]
        state.score_factors = [
            [ScoreFactor.deserialize(f) for f in data["score_factors"][0]],
            [ScoreFactor.deserialize(f) for f in data["score_factors"][1]],
        ]
        state.scores = [
            cast(int, data["scores"][0]),
            cast(int, data["scores"][1]),
        ]
        state.animal_9_moved = data["animal_9_moved"]
        state.ended = data["ended"]
        state.winner = data["winner"]

        return state
