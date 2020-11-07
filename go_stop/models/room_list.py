from typing import Optional
from .room import Room


class RoomList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_to_room: dict = {}

    def find_by_room_id(self, id: str) -> Optional[Room]:
        return next((room for room in self if room.id == id), None)

    def find_by_client_id(self, id: str) -> Optional[Room]:
        try:
            return self.client_to_room[id]
        except KeyError:
            return None

    def make(self, id: str) -> dict:
        if not isinstance(id, str) or id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        if self.find_by_client_id(id) != None:
            return {
                "success": False,
                "error": "The client is already joined in a room",
                "errorCode": 2,
            }

        room = Room()
        room.join(id)
        self.append(room)
        self.client_to_room[id] = room

        return {"success": True, "result": room.id}

    def join(self, id: str, room_id: str) -> dict:
        if not isinstance(id, str) or id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        if self.find_by_client_id(id) != None:
            return {
                "success": False,
                "error": "The client is already joined in a room",
                "errorCode": 2,
            }

        room = self.find_by_room_id(room_id)
        if room == None:
            return {
                "success": False,
                "error": f"There is no room with room id {room.id}",
                "errorCode": 3,
            }

        if len(room.players) == 2:
            return {
                "success": False,
                "error": f"The room {room.id} has 2 people already.",
                "errorCode": 4,
            }

        room.join(id)
        self.client_to_room[id] = room
        return {"success": True}

    def exit(self, id: str) -> dict:
        if not isinstance(id, str) or id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        if self.find_by_client_id(id) == None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        room = self.find_by_client_id(id)
        room.exit(id)
        
        del self.client_to_room[id]

        if len(room.players) == 0:
            self.remove(room)
        
        return {"success": True}

    def start_game(self, id: str) -> dict:
        if not isinstance(id, str) or id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        room = self.find_by_client_id(id)
        if room == None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        if room.game != None:
            return {
                "success": False,
                "error": "The game has been started",
                "errorCode": 3,
            }

        if len(room.players) != 2:
            return {
                "success": False,
                "error": "Not enough players",
                "errorCode": 4,
            }

        room.start_game()
        return {"success": True}

    def end_game(self, id: str) -> dict:
        if not isinstance(id, str) or id == "":
            return {
                "success": False,
                "error": "The client id field is empty.",
                "errorCode": 1,
            }

        room = self.find_by_client_id(id)
        if room == None:
            return {
                "success": False,
                "error": "The client is not in any room",
                "errorCode": 2,
            }

        if room.game == None:
            return {
                "success": False,
                "error": "The game has not been started",
                "errorCode": 3,
            }

        room.end_game()
        return {"success": True}

    def serialize(self):
        return [room.serialize() for room in self]
