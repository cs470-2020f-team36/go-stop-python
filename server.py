from flask_ngrok import _run_ngrok
from flask import Flask, request
from jinja2 import Template
from threading import Timer
import inspect
from flask_socketio import SocketIO, emit
from go_stop.models.room_list import RoomList
from go_stop.models.action import Action


rooms: RoomList = RoomList()


def start_ngrok():
    ngrok_address = _run_ngrok()
    print(ngrok_address)


def run_with_ngrok(app):
    old_run = app.run

    def new_run(*args, **kwargs):
        thread = Timer(1, start_ngrok)
        thread.setDaemon(True)
        thread.start()
        old_run(*args, **kwargs)

    app.run = new_run


app = Flask(__name__)
app.config["SECRET_KEY"] = "" # secret key

# run_with_ngrok(app)
socketio = SocketIO(
    app,
    cors_allowed_origins=[
        # allowed origins
    ],
)


@app.route("/")
def home():
    template = Template("The server is up.")
    return template.render()


@socketio.on("connect")
def on_connect():
    emit(
        "connect response",
        {
            "success": True,
            "message": f"Hello, {request.sid}.",
        },
    )


@socketio.on("list rooms")
def on_list_rooms(msg):
    print(rooms.serialize())
    emit(
        "list rooms response",
        {
            "success": True,
            "result": rooms.serialize(),
        },
    )


@socketio.on("my room")
def on_my_room(msg):
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

        if not game.state.ended:
            game._calculate_scores(without_multiples=True)
        else:
            game._calculate_scores()

        result = game.serialize()
        result.update(
            {
                "actions": [action.serialize() for action in game.actions()],
                "players": room.players,
            }
        )
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
    try:
        res = rooms.find_by_room_id(msg["room"])
        if res is None:
            emit(
                "spectate game response",
                {
                    "success": False,
                    "error": "There is no such a room.",
                    "errorCode": 2,
                },
            )

        if not res.game.state.ended:
            res.game._calculate_scores(without_multiples=True)
        else:
            res.game._calculate_scores()

        result = res.game.serialize()
        result.update(
            {
                "actions": [
                    action.serialize() for action in res.game.actions()
                ],
                "players": res.players,
            }
        )

        emit(
            "spectate game response",
            {
                "success": True,
                "result": result,
            }
            if res != None and res.game != None
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


app.run(host="") # host
