import random
import json
from shortuuid import ShortUUID
from .game import Game


class Room:
    def __init__(self):
        self.id = str(ShortUUID().random(length=4))
        self.players = []
        self.game = None

    def join(self, id: str):
        self.players.append(id)

    def exit(self, id: str):
        self.players.remove(id)
        self.end_game()

    def start_game(self):
        # For testing purpose
        # f = open("./jsons/games/test.json")
        # data = json.load(f)
        # self.game = Game.deserialize(data)

        # Normal play
        random.shuffle(self.players)
        self.game = Game()

    def end_game(self):
        self.game = None

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "players": self.players,
            "gameStarted": self.game != None,
        }
