class Flags:
    def __init__(self):
        self.go = False
        self.select_match = False
        self.shaking = False
        self.move_animal_9 = False
        self.four_of_a_month = False

    def serialize(self) -> dict:
        """
        Serialize the flags.
        """

        return self.__dict__

    @staticmethod
    def deserialize(data: dict):
        """
        Deserialize the flags.
        """

        flags = Flags()
        flags.__dict__ = data

        return flags