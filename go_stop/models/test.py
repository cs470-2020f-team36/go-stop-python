"""
test.py

Execute a unit test.
"""

from typing import List, Tuple, TypeVar
import unittest

from torch import Tensor

from go_stop.models.game import Game
from go_stop.models.player import get_opponent
from go_stop.train.encoder import (
    LEN_EXTENDED_CARDS,
    encode_game,
)
from go_stop.train.sampler import sample_from_observation


Type = TypeVar("Type")


class TestEncoder(unittest.TestCase):
    """Test `go_stop/train/encoder.py`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = Game()

    def test_encode_game(self):
        """Test `encode_game` method."""

        encoded_game = encode_game(self.game, self.game.state.player)

        # current player
        self.assertTrue(encoded_game[[0]].equal(Tensor([0])))

        # opposite hand should be hidden
        hidden_hand = Tensor([0] * (LEN_EXTENDED_CARDS - 1) + [10])
        self.assertTrue(
            encoded_game[
                LEN_EXTENDED_CARDS + 1 : 2 * LEN_EXTENDED_CARDS + 1
            ].equal(hidden_hand)
        )


class TestSampler(unittest.TestCase):
    """Test `go_stop/train/sampler.py`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = Game()

    @staticmethod
    def is_items_identical(lst: List[Tuple[Type]]) -> bool:
        """Check if every item in a list is identical"""
        return tuple(lst) == tuple([lst[0]] * len(lst))

    def test_encode_game(self):
        """Test `sample_from_observation` method."""

        sample = sample_from_observation(self.game, self.game.state.player, 5)

        # Test if it is sampled well.
        # The probability that it will fail is less than 5e-38.
        self.assertFalse(
            TestSampler.is_items_identical(
                [
                    tuple(
                        game.serialize()["board"]["hands"][
                            get_opponent(self.game.state.player)
                        ]
                    )
                    for game in sample
                ]
            )
        )

        # Test if every sample yields the same encoded game.
        self.assertTrue(
            all(
                encode_game(game, game.state.player).equal(
                    encode_game(self.game, game.state.player)
                )
                for game in sample
            )
        )


if __name__ == "__main__":
    unittest.main()
