import random
from shortuuid import ShortUUID
from .game import Game


class Room:
    def __init__(self):
        self.uid = str(ShortUUID().random(length=4))
        self.players = []
        self.game = None

    def join(self, client_id: str):
        self.players.append(client_id)

    def exit(self, client_id: str):
        self.players.remove(client_id)
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
            "id": self.uid,
            "players": self.players,
            "gameStarted": self.game is not None,
        }
