from .card import Card
from .card_list import CardList
from .score_factor import ScoreFactor


class State:
    def __init__(self, player: int = 0):
        """
        State of the game.

        player: 0 | 1
        Index of the current player.

        bomb_increment: int
        Index of the bomb used to guarantee the uniqueness of `index`.

        go_histories: [List[int], List[int]]
        Histories of points at Go's of players.
        For instance, [[8, 10, 32], [9]] means the player 0 claimed Go three times,
        and the player 1 did once, where the points are calculated after applying
        the score advantage of Go's, that is,
            P0: "Go" at 7 -> became 8, "Go" at 9 -> became "10", "Go" at 15 -> became "32" and
            P1: "Go" at 8 -> became 9

                                         card to match, cards to be matched
        select_match: None
                    | [
                        Card,
                        CardList,
                        None | (
                            Card | None,
                            CardList,
                            Card,
                            CardList,
                            CardList,
                            int,
                        )
                    ]
        Save information about which cards the player will select one among.

                card to shake, collection of cards in a shaking
        shaking: None | (Card, CardList (of length >= 3))
        Save information about which cards the player will shake with.

        shaking_histories: [List[CardList], List[CardList]]
        How many shakings are done by players.

        stacking_histories: [Set[int], Set[int]]
        Which stackings are made by players.

        score_factors: [List[ScoreFactor], List[ScoreFactor]]
        Sets of score factors, which are updated at almost end of each turn.
        ScoreFactor of kind "go" and "penalties" should be at the very last.

        scores: [int, int]

        animal_9_moved: bool | None
        Whether the animal card of September has moved to the junk field.
        `None` if it is not queried to move the animal card of September.

        ended: bool
        Whether the game ended.

        winner: None | 0 | 1
        Who won the game, if it is ended.
        """

        self.starting_player = player
        self.player = player
        self.bomb_increment = 0
        self.go_histories = [[], []]
        self.select_match = None
        self.shaking = None
        self.shaking_histories = [[], []]
        self.stacking_histories = [set(), set()]
        self.score_factors = [[], []]
        self.scores = [0, 0]
        self.animal_9_moved = None
        self.ended = False
        self.winner = None

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
            if self.select_match == None
            else [
                self.select_match[0].serialize(),
                None
                if self.select_match[1] is None
                else self.select_match[1].serialize(),
                None
                if self.select_match[2] is None
                else [
                    None
                    if self.select_match[2][0] == None
                    else self.select_match[2][0].serialize(),
                    self.select_match[2][1].serialize(),
                    self.select_match[2][2].serialize(),
                    self.select_match[2][3].serialize(),
                    self.select_match[2][4],
                ],
            ],
            "shaking": None
            if self.shaking == None
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
        state.go_histories = data["go_histories"]
        state.select_match = (
            None
            if data["select_match"] == None
            else [
                Card.deserialize(data["select_match"][0]),
                CardList.deserialize(data["select_match"][1]),
                None
                if data["select_match"][2] == None
                else (
                    None
                    if data["select_match"][2][0] == None
                    else Card.deserialize(data["select_match"][2][0]),
                    CardList.deserialize(data["select_match"][2][1]),
                    Card.deserialize(data["select_match"][2][2]),
                    CardList.deserialize(data["select_match"][2][3]),
                    data["select_match"][2][4],
                ),
            ]
        )
        state.shaking = (
            None
            if data["shaking"] == None
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
        state.scores = data["scores"]
        state.animal_9_moved = data["animal_9_moved"]
        state.ended = data["ended"]
        state.winner = data["winner"]

        return state