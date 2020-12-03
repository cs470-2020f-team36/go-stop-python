"""
setting.py

Define some game settings.
"""

# pylint: disable=too-many-instance-attributes, too-few-public-methods


class Setting:
    """Go-Stop game settings."""

    def __init__(self):
        self.junk_card_from_ttadak = 1
        self.junk_card_from_discard_and_match = 1
        self.junk_card_from_clear = 1
        self.junk_card_from_bomb = 1
        self.junk_card_from_stacking = 1
        self.junk_card_from_self_stacking = 2
        self.junk_card_from_bonus_card = 1

        self.score_of_four_of_a_month = 10
        self.score_of_three_stackings = 10

        self.proceed_when_center_field_has_a_four_of_a_month = False
