"""
room_list.py

A class for a list of rooms with a bunch of methods.
"""

from typing import Dict, Optional

from .room import Room


class RoomList(list):
    """RoomList class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_to_room: Dict[str, Optional[Room]] = {}

    def find_by_room_id(self, room_id: str) -> Optional[Room]:
        """Find a room by `room_id`."""
        return next((room for room in self if room.uid == room_id), None)

    def find_by_client_id(self, client_id: str) -> Optional[Room]:
        """Find a room by `client_id`."""
        try:
            return self.client_to_room[client_id]
        except KeyError:
            return None

    def make(self, client_id: str) -> dict:
        """Make a room and put the client `client_id` in the room."""
        if not isinstance(client_id, str) or client_id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        if self.find_by_client_id(client_id) is not None:
            return {
                "success": False,
                "error": "The client is already joined in a room",
                "errorCode": 2,
            }

        room = Room()
        room.join(client_id)
        self.append(room)
        self.client_to_room[client_id] = room

        return {"success": True, "result": room.serialize()}

    def join(self, client_id: str, room_id: str) -> dict:
        """Let the client `client_id` join the room `room_id`."""
        if not isinstance(client_id, str) or client_id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        if self.find_by_client_id(client_id) is not None:
            return {
                "success": False,
                "error": "The client is already joined in a room",
                "errorCode": 2,
            }

        room = self.find_by_room_id(room_id)
        if room is None:
            return {
                "success": False,
                "error": f"There is no room with room id {room_id}",
                "errorCode": 3,
            }

        if len(room.players) == 2:
            return {
                "success": False,
                "error": f"The room {room.uid} has 2 people already.",
                "errorCode": 4,
            }

        if room.game is not None:
            return {
                "success": False,
                "error": f"The game has stared in the room {room.uid}.",
                "errorCode": 5,
            }

        room.join(client_id)
        self.client_to_room[client_id] = room
        return {"success": True}

    def exit(self, client_id: str) -> dict:
        """Let the client `client_id` exit its room."""
        if not isinstance(client_id, str) or client_id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        room = self.find_by_client_id(client_id)

        if room is None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        room.exit(client_id)

        del self.client_to_room[client_id]

        if len(room.players) == 0:
            self.remove(room)

        return {"success": True}

    def start_game(self, client_id: str) -> dict:
        """Let the client `client_id` start a game in its room."""
        if not isinstance(client_id, str) or client_id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        room = self.find_by_client_id(client_id)
        if room is None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        if room.game is not None:
            return {
                "success": False,
                "error": "The game has been started",
                "errorCode": 3,
            }

        if len(room.players) != 2 and client_id != "kanu":
            return {
                "success": False,
                "error": "Competing with AlphaGoStop is not availble yet",
                "errorCode": 4,
            }

        if len(room.players) != 2:
            room.start_game(single_player=True)

        else:
            room.start_game()
        return {"success": True}

    def end_game(self, client_id: str) -> dict:
        """Let the client `client_id` end the current game in its room."""
        if not isinstance(client_id, str) or client_id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        room = self.find_by_client_id(client_id)
        if room is None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        if room.game is None:
            return {
                "success": False,
                "error": "The game has not been started",
                "errorCode": 3,
            }

        room.end_game()
        return {"success": True}

    def serialize(self):
        """Serialize a RoomList."""
        return [room.serialize() for room in self]
