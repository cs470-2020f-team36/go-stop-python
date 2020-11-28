import json

from go_stop.models.action import ActionThrow, ActionSelectMatch
from go_stop.models.card import AnimalCard, BonusCard, BrightCard, JunkCard
from go_stop.models.game import Game


def show_separator():
    print("------------")


# create a new game
# game = Game()

# or load from a json file (Refer ./jsons/games/test.json)
f = open("./jsons/games/test.json")
data = json.load(f)
game = Game.deserialize(data)
f.close()

# print the serialized form of the game
print(game.serialize())

show_separator()

# You may see the list of actions that
# the current player (game.state.player)
# is able to do via `game.actions()`.
def show_actions():
    print([action.serialize() for action in game.actions()])


show_actions()

show_separator()

# The current player commits the action
# via the method game.play(action: Action)
game.play(ActionThrow(JunkCard(month=6, index=1)))

show_separator()

# Now, the current player is changed.
print(game.serialize())

show_separator()

# There are more kinds of actions that the player can do.
show_actions()

show_separator()

# The player can throw a bonus card.
# This will take junk cards from the opponent.
game.play(ActionThrow(BonusCard(2)))
game.play(ActionThrow(BonusCard(3)))

show_separator()

# Note that there are stackings of three cards of same months.
print(
    dict(
        (month, cards.serialize())
        for (month, cards) in game.board.center_field.items()
    )
)

show_separator()

# By taking those, the player can take another junk from the opponent.
game.play(ActionThrow(BrightCard(11)))

# The log shows two `TRIPLE MATCHES`es and one `CLEAR` (쓸).
# This should take 3 junk cards from the opponent,
# but the opponent only has 2 of them.
# So the player took all of them.

show_separator()

print(game.board.serialize())

show_separator()
show_separator()
show_separator()

# second game (test select match and ttadak)
f = open("./jsons/games/test2.json")
data = json.load(f)
game = Game.deserialize(data)
f.close()

print(game.serialize())
show_separator()
show_actions()
show_separator()

# throw J091
game.play(ActionThrow(JunkCard(9, 1)))

# there are A09 and R09 on the center_field, so the player should decide whether one will be captured
show_actions()
show_separator()

# choose to capture A09
game.play(ActionSelectMatch(AnimalCard(9)))

# the player flip +2 and J090, which is *ttadak*
print(game.serialize()["logs"])
show_separator()

print(game.serialize())

# third game (test stacking)
f = open("./jsons/games/test3.json")
data = json.load(f)
game = Game.deserialize(data)
f.close()

print(game.serialize())
show_separator()
show_actions()
show_separator()

# throw J091
game.play(ActionThrow(JunkCard(9, 1)))

# there is A09 on the center_field (single match)
# the player flip +2 and J090, which is *stacking*
# so the center_field now has a stacking of J091, A09, +2, and J090
print(game.serialize()["logs"])
show_separator()

print(game.serialize())


# For more information, see models/ and constants/ directories.
# Also, you may run the server with the file `server.py`.
