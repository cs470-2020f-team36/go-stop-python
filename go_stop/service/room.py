"""
room.py

A class for a room for the clients to play a Go-Stop game.
"""

import os
import random
from typing import Optional

from shortuuid import ShortUUID

from ..models.game import Game
from ..service.ai import ai, estimate


class Room:
    """Room class."""

    def __init__(self):
        self.uid = str(ShortUUID().random(length=4))
        self.players = []
        self.game = None
        self.single_player = False

    def num_of_players(self):
        """The number of human players."""
        if os.environ["AI_AGENT_ID"] in self.players:
            return len(self.players) - 1
        return len(self.players)

    def join(self, client_id: str):
        """Let the client `client_id` join the room."""
        self.players.append(client_id)

    def exit(self, client_id: str):
        """Let the client `client_id` exit the room."""
        if os.environ["AI_AGENT_ID"] in self.players:
            self.players.remove(os.environ["AI_AGENT_ID"])
        self.players.remove(client_id)
        self.end_game()

    def start_game(self, single_player=False):
        """Let the client `client_id` start a game in the room."""
        if os.environ["AI_AGENT_ID"] not in self.players:
            self.players.append(os.environ["AI_AGENT_ID"])

        random.shuffle(self.players)

        game = Game()
        self.game = game
        self.single_player = single_player

        # If the room is a single-player room and the first player is the AI agent,
        # let the AI play actions.
        if single_player:
            while game.state.player == self.players.index(
                os.environ["AI_AGENT_ID"]
            ):
                action = ai.query(game)
                game.play(action)

        # Calculate intermediate/final scores.
        if not game.state.ended:
            game.calculate_scores(without_multiples=True)
        else:
            game.calculate_scores()

    def end_game(self):
        """Let the client `client_id` end a game in the room."""
        if os.environ["AI_AGENT_ID"] in self.players:
            self.players.remove(os.environ["AI_AGENT_ID"])

        self.game = None

    def serialize(self) -> dict:
        """Serialize a room."""
        return {
            "id": self.uid,
            "players": [
                p for p in self.players if p != os.environ["AI_AGENT_ID"]
            ],
            "gameStarted": self.game is not None,
            "singlePlayer": self.single_player,
        }

    def serialize_game(self) -> Optional[dict]:
        """Serialize game if the game is presented in this room."""
        if self.game is None:
            return None

        result = self.game.serialize()
        result.update(
            {
                "actions": [
                    action.serialize() for action in self.game.actions()
                ],
                "players": self.players,
            }
        )

        if self.single_player and not self.game.state.ended:
            ai_index = self.players.index(os.environ["AI_AGENT_ID"])
            estimated_result = estimate(self.game, ai_index)
            result.update({"estimate": list(estimated_result)})
        else:
            result.update({"estimate": None})

        return result
