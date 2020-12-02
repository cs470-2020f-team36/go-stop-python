"""
app.py

The server instance implemented on flask_socketio.
"""


import os
import time

from flask import Flask, request
from flask_socketio import SocketIO, emit

from go_stop.models.action import Action
from go_stop.service.ai import ai
from go_stop.service.room_list import RoomList


rooms: RoomList = RoomList()


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ["APP_SECRET_KEY"]

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
)


@socketio.on("connect")
def on_connect():
    """On connect."""
    emit(
        "connect response",
        {
            "success": True,
            "message": f"Hello, {request.sid}.",
        },
    )


@socketio.on("list rooms")
def on_list_rooms(_):
    """When the client requested the list of presented rooms."""
    emit(
        "list rooms response",
        {
            "success": True,
            "result": rooms.serialize(),
        },
    )


@socketio.on("my room")
def on_my_room(msg):
    """When the client requested its room."""
    try:
        if not isinstance(msg["client"], str) or msg["client"] == "":
            emit(
                "my room response",
                {
                    "success": False,
                    "error": "The client id field is empty.",
                    "errorCode": 1,
                },
            )
            return

        res = rooms.find_by_client_id(msg["client"])

        emit(
            "my room response",
            {
                "success": True,
                "result": res.serialize() if res is not None else None,
            },
        )

    except KeyError:
        emit(
            "my room response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("make room")
def on_make_room(msg):
    """When the client requested to make a room."""
    try:
        res = rooms.make(msg["client"])

        emit("make room response", res)

        if res["success"]:
            emit(
                "list rooms response",
                {
                    "success": True,
                    "result": rooms.serialize(),
                },
                broadcast=True,
            )

    except KeyError:
        emit(
            "make room response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("join room")
def on_join_room(msg):
    """When the client requested to join a room."""
    try:
        res = rooms.join(msg["client"], msg["room"])

        emit("join room response", res)

        if res["success"]:
            emit(
                "list rooms response",
                {
                    "success": True,
                    "result": rooms.serialize(),
                },
                broadcast=True,
            )

    except KeyError:
        emit(
            "join room response",
            {
                "success": False,
                "error": "The client id field or the room id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("exit room")
def on_exit_room(msg):
    """When the client requested to exit its room."""
    try:
        res = rooms.exit(msg["client"])

        emit("exit room response", res)

        if res["success"]:
            emit(
                "list rooms response",
                {
                    "success": True,
                    "result": rooms.serialize(),
                },
                broadcast=True,
            )

    except KeyError:
        emit(
            "exit room response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("play")
def on_play(msg):
    """When the client requested to play an action."""
    try:
        player = msg["client"]
        action = Action.deserialize(msg["action"])

        room = rooms.find_by_client_id(msg["client"])

        if room is None:
            emit(
                "play response",
                {
                    "success": False,
                    "error": "The client is not in any room",
                    "errorCode": 2,
                },
            )
            return

        game = room.game

        if room.players[game.state.player] != player:
            emit(
                "play response",
                {
                    "success": False,
                    "error": "Not your turn",
                    "errorCode": 3,
                },
            )
            return

        res = game.play(action)

        if not res:
            emit(
                "play response",
                {
                    "success": False,
                    "error": "Invalid action",
                    "errorCode": 4,
                },
            )
            return

        if room.single_player:
            while (
                game.state.player
                == room.players.index(os.environ["AI_AGENT_ID"])
                and not game.state.ended
            ):
                result = room.serialize_game()
                emit(
                    "spectate game response",
                    {
                        "success": True,
                        "result": result,
                    },
                    broadcast=True,
                )

                action = ai.query(game)
                game.play(action)
                time.sleep(2)

                result = room.serialize_game()
                emit(
                    "spectate game response",
                    {
                        "success": True,
                        "result": result,
                    },
                    broadcast=True,
                )

        if not game.state.ended:
            game.calculate_scores(without_multiples=True)
        else:
            game.calculate_scores()

        result = room.serialize_game(set_estimate=False)
        emit(
            "play response",
            {
                "success": True,
                "result": result,
            },
        )
        emit(
            "spectate game response",
            {
                "success": True,
                "result": result,
            },
            broadcast=True,
        )

    except KeyError:
        emit(
            "exit room response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("start game")
def on_start_game(msg):
    """When the client requested to start a game in its room."""
    try:
        res = rooms.start_game(msg["client"])

        emit("start game response", res)

        if res["success"]:
            emit(
                "list rooms response",
                {
                    "success": True,
                    "result": rooms.serialize(),
                },
                broadcast=True,
            )

    except KeyError:
        emit(
            "start game response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("end game")
def on_end_game(msg):
    """When the client requested to end the game in its room."""
    try:
        res = rooms.end_game(msg["client"])

        emit("end game response", res)

        if res["success"]:
            emit(
                "list rooms response",
                {
                    "success": True,
                    "result": rooms.serialize(),
                },
                broadcast=True,
            )

    except KeyError:
        emit(
            "end game response",
            {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            },
        )


@socketio.on("spectate game")
def on_spectate_game(msg):
    """When the client requested to spectate a game."""
    try:
        room = rooms.find_by_room_id(msg["room"])
        if room is None:
            emit(
                "spectate game response",
                {
                    "success": False,
                    "error": "There is no such a room.",
                    "errorCode": 2,
                },
            )
            return

        if not room.game.state.ended:
            room.game.calculate_scores(without_multiples=True)
        else:
            room.game.calculate_scores()

        result = room.serialize_game(set_estimate=False)

        emit(
            "spectate game response",
            {
                "success": True,
                "result": result,
            }
            if room.game is not None
            else {
                "success": False,
                "error": "The game has not been started in the room.",
                "errorCode": 2,
            },
        )

    except KeyError:
        emit(
            "spectate game response",
            {
                "success": False,
                "error": "The room id field is empty.",
                "errorCode": 1,
            },
        )


if __name__ == "__main__":
    app.run()  # host
