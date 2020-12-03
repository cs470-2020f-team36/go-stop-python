"""
game.py

Implement the main Go-Stop game class.
"""

import copy
from itertools import groupby
from typing import Any, List, Optional, Tuple, cast

from ..constants.card import go_stop_cards
from .action import (
    Action,
    ActionThrow,
    ActionThrowBomb,
    ActionBomb,
    ActionShakable,
    ActionShaking,
    ActionSelectMatch,
    ActionFourOfAMonth,
    ActionGo,
    ActionMoveAnimal9,
)
from .board import Board
from .card import (
    Card,
    BrightCard,
    AnimalCard,
    RibbonCard,
    BonusCard,
    SpecialCard,
)
from .card_list import CardList
from .flags import Flags
from .logger import Logger
from .player import Player, get_opponent
from .scorer import calculate_score
from .score_factor import ScoreFactor
from .setting import Setting
from .state import State


# pylint: disable=too-many-lines, too-many-arguments, too-many-return-statements
# pylint: disable=too-many-locals, too-many-branches, too-many-statements


class Game(Setting):
    """Go-Stop game class."""

    def __init__(self, starting_player: Player = 0):
        super().__init__()

        self.board = Board(starting_player)
        self.state = State(starting_player)
        self.flags = Flags()
        self.logger = Logger()

        four_of_a_month = self.board.four_of_a_month()

        # The list of players (0, 1) having a four-of-a-month.
        players_having_four_of_a_month = [
            cast(Player, player)
            for player in [0, 1]
            if len(four_of_a_month[player]) > 0
        ]

        # Due to self.board._is_reset_necessary(),
        # players_having_four_of_a_month will cannot have 2 players.
        assert len(players_having_four_of_a_month) < 2

        if players_having_four_of_a_month != []:
            self.state.player = players_having_four_of_a_month[0]
            self.flags.four_of_a_month = True

    def actions(self) -> List[Action]:
        """Returns a list of possible actions with this game"""

        board = self.board
        state = self.state
        flags = self.flags

        # If the game is over, return the empty list.
        if state.ended:
            return []

        hand = board.hands[state.player]

        # If a four-of-a-month flag is set,
        # return actions to choose whether to finish the game or not.
        if flags.four_of_a_month:
            return [ActionFourOfAMonth(option) for option in {True, False}]

        # If a Go flag is set, return actions to choose whether to declare Go or not.
        if flags.go:
            return [ActionGo(option) for option in {True, False}]

        # If a select match flag is set, return actions to choose which cards will be selected.
        if flags.select_match:
            assert state.select_match is not None
            matches: CardList = state.select_match[1]
            return [ActionSelectMatch(match) for match in matches]

        # If a shaking flag is set,
        # return actions to choose whether the cards are going to be shaked or not.
        if flags.shaking:
            return [ActionShaking(option) for option in {True, False}]

        # If a move animal 9 flag is set, return actions to choose whether
        # the animal of September card will be used as a double junk (True)
        # or as an animal (False).
        if flags.move_animal_9:
            return [ActionMoveAnimal9(option) for option in {True, False}]

        # Now, none of the flags are set.

        # Gather the months of cards in the hand.
        months_in_hand = [card.month for card in hand if card.month is not None]
        months_in_hand.sort()
        center_field_months = [
            month for month in range(1, 13) if board.center_field[month] != []
        ]

        # If a player has 3 or more cards sharing the month,
        # and the center field does not have any card of that month,
        # they could be shaken.
        shakable_months = [
            month
            for month, group in groupby(months_in_hand)
            if len(list(group)) >= 3 and month not in center_field_months
        ]

        # If either a player has 3 cards sharing the month
        # and the center field has a remaining card of that month,
        # or a player has 2 cards sharing the month
        # and the center field has remaining 2 cards of that month,
        # the cards in the hand should be used as a bomb.
        #
        # To gather the months of which cards in the hand can be used as a bomb,
        # we first combine the hand and the center field.
        combined_hand = hand + [
            item for sublist in board.center_field.values() for item in sublist
        ]
        months_in_combined_hand = [
            card.month for card in combined_hand if card.month is not None
        ]
        months_in_combined_hand.sort()

        # Now, the `bomb_months` is the list of months
        # of which there are more than 2 cards in the hand
        # and there are 4 cards in the `combined_hand`.
        bomb_months = [
            month
            for month, group in groupby(months_in_combined_hand)
            if len(list(group)) == 4 and len(hand.of_month(month)) >= 2
        ]

        return (
            cast(
                List[Action],
                [
                    ActionThrow(card)
                    for card in hand
                    if card.month not in shakable_months + bomb_months
                    and card.kind != "bomb"
                ],
            )
            + ([ActionThrowBomb()] if any(card.kind == "bomb" for card in hand) else [])
            + [
                ActionShakable(card)
                for month in shakable_months
                for card in hand
                if card.month == month
            ]
            + [ActionBomb(month) for month in bomb_months]
        )

    def play(self, action: Action, pass_check=False) -> bool:
        """
        Play the given action on this game, and returns whether it succeeded.

        `pass_check` is whether to pass checking `action not in self.actions()`.
        It is needed to handle `ActionShaking` using `ActionThrow`.
        """

        if not pass_check and action not in self.actions():
            return False

        board = self.board
        state = self.state
        flags = self.flags

        if action.kind in {"throw", "throw bomb", "bomb", "select match"}:
            return self._play_action_of_kind_throw_throwbomb_bomb_selectmatch(action)

        if action.kind == "shaking":
            action = cast(ActionShaking, action)

            # `state.shaking` should not be None
            assert state.shaking is not None

            # `card` is the card to throw, and `cards` is the list of cards that will be shaken.
            card, cards = cast(Tuple[Card, CardList], state.shaking)

            # Reset the shaking state and flag.
            flags.shaking = False
            state.shaking = None

            if action.option:
                # If the player selected to shake,
                # add the month of the cards into the shaking history.
                state.shaking_histories[state.player].append(CardList(cards))
                self.logger.log("shaking", CardList(cards))

            # The remaining process is the same with throwing `card`.
            return self.play(ActionThrow(card), pass_check=True)

        if action.kind == "shakable":
            action = cast(ActionShakable, action)

            # Special cards and bonus cards are not shakable.
            assert action.card.month is not None

            # Set the shaking state and flag.
            flags.shaking = True
            state.shaking = (
                action.card,
                board.hands[state.player].of_month(action.card.month),
            )

            # Wait for the response from the player.
            return True

        if action.kind == "four of a month":
            action = cast(ActionFourOfAMonth, action)

            # Reset the four-of-a-month flag.
            flags.four_of_a_month = False

            if action.option:
                # The player chose to finish the game.
                state.ended = True
                state.winner = state.player
                state.score_factors[state.player] = [ScoreFactor("four of a month")]
                self.logger.log(
                    "four of a month",
                    go_stop_cards.of_month(
                        list(board.four_of_a_month()[state.player])[0]
                    ),
                )

                return True

            # Change the turn to the starting player and resume the game.
            if state.player != state.starting_player:
                self.logger.log("turn change", (state.player, state.starting_player))
                state.player = state.starting_player

            return True

        if action.kind == "go":
            action = cast(ActionGo, action)

            # Reset the Go flag.
            flags.go = False

            if action.option:
                # Go
                self._go()
                return True

            # Stop
            self._stop()
            return True

        assert action.kind == "move animal 9"
        action = cast(ActionMoveAnimal9, action)

        # Reset the move animal 9 flag.
        flags.move_animal_9 = False

        # Set `state.animal_9_moved`.
        state.animal_9_moved = action.option
        self.logger.log("move animal 9", action.option)

        # Recalculate the scores without multiples (since the game is not over).
        self.calculate_scores(without_multiples=True)

        # Check if the player can declare Go/Stop.
        if (
            state.go_histories[state.player] == []
            and self.state.scores[state.player] >= 7
        ) or (
            state.go_histories[state.player] != []
            and self.state.scores[state.player] > state.go_histories[state.player][-1]
        ):
            if board.hands[state.player] == []:
                # If there are no cards anymore in the hand, declare Stop automatically.
                self._stop()
                return True

            # Set the Go flag to True to wait for the player's response.
            flags.go = True
            return True

        # If the player cannot declare Go/Stop, change the turn.
        self.logger.log("turn change", (state.player, get_opponent(state.player)))
        state.player = get_opponent(state.player)

        # Check for the push back. (Both players did not win.)
        if (
            state.player == state.starting_player
            and board.hands[state.starting_player] == []
        ):
            state.ended = True
            state.winner = None

        return True

    def _play_action_of_kind_throw_throwbomb_bomb_selectmatch(
        self, action: Action
    ) -> bool:
        """
        Play an action whose kind is one of the following:
        "throw", "throw bomb", "bomb", or "select match".
        """

        board = self.board
        state = self.state
        flags = self.flags

        # handles events on "throw" until flipping
        if action.kind in {"throw", "throw bomb"}:
            # If the action kind is "throw bomb", regard it as throwing an actual bomb card.
            if action.kind == "throw bomb":
                card_thrown: Card = next(
                    (c for c in board.hands[state.player] if c.kind == "bomb"),
                    cast(Any, None),
                )
                action = ActionThrow(card_thrown)

            # Assert that now the action becomes an instance of `ActionThrow`.
            action = cast(ActionThrow, action)

            # Remove the card thrown from the hand.
            board.hands[state.player].remove(action.card)
            self.logger.log("throw", action.card)

            # if the player threw a bonus card
            if action.card.kind == "bonus":
                # append it to the capture field
                self._append_to_capture_field(CardList([action.card]))

                # take junk from the opponent
                self._take_junk_from_opponent(self.junk_card_from_bonus_card)

                # append the flipped card to the hand
                card_flipped = self._flip_card()
                self._append_to_hand(CardList([card_flipped]))
                self.calculate_scores(without_multiples=True)

                return True

            # elsewise, throw a card (or a bomb)
            # `captures_before` is the list of cards which were captured before flipping,
            # and `junk_count_before` is the number of junk cards
            # which were taken from the opponent before flipping.
            (captures_before, junk_count_before) = self._throw_card(
                action.card, action.card
            )

            # if it signals to terminate `self.play`, terminate it.
            # (Refer to `self._throw_card` docstring.)
            if captures_before is None:
                return True

            # Flip the top card from the drawing pile, and throw it.
            (
                flag_continue,
                card_flipped,
                bonus_captures,
                captures_after,
                junk_count,
            ) = self._flip_and_then_throw(
                action.card, captures_before, junk_count_before
            )

            # If `flag_continue` is set to False, return True.
            if not flag_continue:
                return True

            # `action.card` is the card thrown. (Note that `action` was an `ActionThrow`.)
            card_thrown = action.card

        # handles events on "bomb" until flipping
        elif action.kind == "bomb":
            # `month` is the (common) month of cards that will be thrown as a bomb.
            month = action.arg
            self.logger.log("bomb", month)

            # if the player throw a bomb, take one junk card from the opponent
            junk_count_before = self.junk_card_from_bomb

            # and captures all four cards
            captures_before = CardList(
                [card for card in board.hands[state.player] if card.month == month]
                + board.center_field[month]
            )

            # Remove all the card used.
            bomb_number = len(board.hands[state.player].of_month(month)) - 1
            bombed_cards = board.hands[state.player].of_month(month)
            board.hands[state.player] = board.hands[state.player].except_month(month)

            # empty the center field of corresponding month
            board.center_field[month] = CardList()

            # Append the bomb cards into the hand, and increase `bomb_increment`.
            # Note that `bomb_increment` is used to ensure the uniqueness of cards
            # in the hand.
            for _ in range(bomb_number):
                board.hands[state.player].append(
                    SpecialCard("bomb", state.bomb_increment)
                )
                state.bomb_increment += 1

            # Throwing a bomb is treated as a shaking also.
            # Append the month to the shaking history.
            state.shaking_histories[state.player].append(bombed_cards)

            # Flip the top card from the drawing pile, and throw it.
            (
                flag_continue,
                card_flipped,
                bonus_captures,
                captures_after,
                junk_count,
            ) = self._flip_and_then_throw(None, captures_before, junk_count_before)

            # If `flag_continue` is set to False, return True.
            if not flag_continue:
                return True

            # `action.card` is set to None.
            card_thrown = None

        elif action.kind == "select match":
            action = cast(ActionSelectMatch, action)

            # When the select match flag is set, `state.select_match` should not be None.
            assert state.select_match is not None

            card_thrown, matches, arg_after = cast(
                Tuple[Card, CardList, Any], state.select_match
            )

            # The selected match should be in `matches`.
            assert action.match in matches

            if arg_after is None:
                # If `arg_after` is None,
                # it means the action was done before flipping the top card from the drawing pile.
                assert card_thrown.month is not None

                # The cards captured (before the flipping) are `card_thrown` and `action.match`.
                captures_before = CardList([card_thrown, action.match])
                self.logger.log("select match", (card_thrown, action.match))

                # Remove the matched card.
                board.center_field[card_thrown.month].remove(action.match)

                # Reset the `select_match` state and flag.
                state.select_match = None
                flags.select_match = False

                # Flip the top card from the drawing pile, and throw it.
                (
                    flag_continue,
                    card_flipped,
                    bonus_captures,
                    captures_after,
                    junk_count,
                ) = self._flip_and_then_throw(card_thrown, captures_before, 0)

                # If `flag_continue` is set to False, return True.
                if not flag_continue:
                    return True

            else:
                # After flipping the top card from the drawing pile.

                # `arg_after` should not be None
                assert arg_after is not None

                # `card_flipped` and `action.match` will be matched.
                (
                    captures_before,
                    card_flipped,
                    bonus_captures,
                    junk_count,
                ) = arg_after
                captures_after = CardList([card_flipped, action.match])
                self.logger.log("select match", (card_flipped, action.match))

                assert card_flipped.month is not None
                board.center_field[card_flipped.month].remove(action.match)

                state.select_match = None
                flags.select_match = False

        # Assert that `captures_before` and `captures_after` are CardLists.
        captures_before = cast(CardList, captures_before)
        captures_after = cast(CardList, captures_after)

        # Check for discard-and-match, ttadak, stacking, or clear; and tune the junk count.
        # Refer to the method docstring.

        # Note that the `card_thrown` argument is None
        # if action.kind is either "throw bomb" or "bomb",
        # since, for the former case, the player actually did not throw any card,
        # and for the latter case, there are no need to check for
        # discard-and-match, ttadak, stacking, or clear.
        junk_count += self._check_after_flip(
            card_thrown if action.kind in {"throw", "select match"} else None,
            captures_before,
            card_flipped,
            bonus_captures,
            captures_after,
        )

        # take junks
        self._take_junk_from_opponent(junk_count)

        # calculate the score factors
        self.calculate_scores(without_multiples=True)

        # check move_animal_9
        if (
            AnimalCard(9) in board.capture_fields[state.player]
            and state.animal_9_moved is None
        ):
            # first, check the score of the player
            state.animal_9_moved = True
            self.calculate_scores(without_multiples=True)
            score_after_animal_9 = state.scores[state.player]

            state.animal_9_moved = None
            self.calculate_scores(without_multiples=True)

            if score_after_animal_9 >= 7:
                flags.move_animal_9 = True
                return True

        # check go
        if (
            state.go_histories[state.player] == []
            and calculate_score(state, state.player) >= 7
        ) or (
            state.go_histories[state.player] != []
            and calculate_score(state, state.player)
            > state.go_histories[state.player][-1]
        ):
            if board.hands[state.player] == []:
                self._stop()
                return True

            else:
                flags.go = True
                return True

        # pass the turn
        self.logger.log("turn change", (state.player, get_opponent(state.player)))
        state.player = get_opponent(state.player)

        # Check for the push back. (Both players did not win.)
        if (
            state.player == state.starting_player
            and board.hands[state.starting_player] == []
        ):
            state.ended = True
            state.winner = None

        return True

    def _append_to_hand(self, cards: CardList) -> None:
        """Append cards to the player's hand."""

        self.board.hands[self.state.player].extend(cards)
        self.logger.log("append to hand", (self.state.player, CardList(cards)))
        self.sort_board()

    def _append_to_center_field(self, cards: CardList) -> None:
        """
        Discard cards onto the center field.
        Here, cards should have the same months, except for bonus cards.
        """

        if cards == []:
            return

        # cards should have the same month except bonus cards
        assert len(set(card.month for card in cards if card.month is not None)) == 1
        month = cards[0].month

        self.board.center_field[month].extend(cards)
        self.logger.log("append to center field", CardList(cards))
        self.sort_board()

    def _append_to_capture_field(self, cards: CardList) -> None:
        """Append cards to the player's capture field."""
        self.board.capture_fields[self.state.player].extend(cards)
        self.logger.log("append to capture field", (self.state.player, CardList(cards)))

    def sort_board(self) -> None:
        """Sorts the hands and the center field of the board."""
        self.board.sort()

    def _throw_card(
        self, card: Card, card_thrown_before_flipping: Card
    ) -> Tuple[Optional[CardList], int]:
        """
        Throw a card (or a bomb), and returns
        (captures: CardList | None, junk_count: int).

        `card` is the card that will be thrown now.
        Note that the card flipped from the drawing pile will also be treated
        by `_throw_card` method, and in this case, the `card` argument
        will indicate the flipped card.

        `card_thrown_before_flipping` is the card that was thrown
        before the flipping (if occurred.)
        If the `_throw_card` is called before the flipping,
        `card` and `card_thrown_before_flipping` are equal.
        This argument is required while setting state.select_match.

        If captures is None, it means
        `self.play(action)` should be terminated (with some flags set).

        `junk_count` means the number of junks taken from the opponent.
        """

        board = self.board
        state = self.state
        flags = self.flags

        captures = CardList()
        junk_count = 0

        # By throwing nothing, the player cannot capture anything.
        if card.kind == "bomb":
            return (captures, 0)

        # Bonus cards and bomb cards are handled beforehand.
        # Thus, `card.month` should not be None
        assert card.month is not None

        # The list of cards in the center field of the month `card.month`.
        center_field_of_month = copy.copy(board.center_field[card.month])

        # If `center_field_of_month` is empty, append the card thrown to the center field.
        if center_field_of_month == []:
            self._append_to_center_field(CardList([card]))

        # If `center_field_of_month` is a singleton, a single match occurs.
        elif len(center_field_of_month) == 1:
            captures.extend([card, center_field_of_month[0]])
            self.logger.log("single match", (card, center_field_of_month[0]))
            board.center_field[card.month].remove(center_field_of_month[0])

        # If `center_field_of_month` is a doubleton, a "double matches" occurs.
        # In this case, select match state and flag are set
        # unless the cards are junk cards with the same month and the same multiple.
        elif len(center_field_of_month) == 2:
            # Check if two matches are all junk cards of the same multiple:
            if (
                center_field_of_month[0].kind == center_field_of_month[1].kind == "junk"
                and center_field_of_month[0].multiple
                == center_field_of_month[1].multiple
            ):
                captures.extend([card, center_field_of_month[0]])
                self.logger.log("double matches", (card, center_field_of_month))
                board.center_field[card.month].remove(center_field_of_month[0])

            # else, set the select match flag
            else:
                flags.select_match = True
                state.select_match = (
                    card_thrown_before_flipping,
                    center_field_of_month,
                    None,
                )
                self.logger.log("double matches", (card, center_field_of_month))
                return (None, 0)

        # If `center_field_of_month` is a stacking, a "triple matches" occurs.
        # In this case, the player will take a junk from the opponent.
        # Note that there could be (one or two) bonus cards in the center field,
        # and we need to add the junk counts from those bonus cards.
        elif len(center_field_of_month) >= 3:
            # Increase the junk count from capturing a stacking
            junk_count += (
                self.junk_card_from_self_stacking
                if card.month in state.stacking_histories[state.player]
                else self.junk_card_from_stacking
            )
            # Increase the junk count from capturing bonus cards.
            junk_count += self.junk_card_from_bonus_card * (
                len(board.center_field[card.month]) - 3
            )

            # Extend `captures` with captured cards.
            captures.extend([card] + center_field_of_month)

            self.logger.log("triple matches", (card, center_field_of_month))

            board.center_field[card.month] = CardList()

        else:
            raise Exception("Something wrong!")

        return (captures, junk_count)

    def _flip_card(self) -> Card:
        """Flip the top card from the drawing pile and return it."""
        card = self.board.drawing_pile.pop(0)
        self.logger.log("flip", card)
        return card

    def _flip_card_until_normal(self) -> Tuple[Card, CardList, int]:
        """Repeat to flip the top card from the drawing pile until it is not a bonus card."""
        flipped = self._flip_card()
        bonus_captures = CardList()
        junk_count = 0

        while flipped.kind == "bonus":
            bonus_captures.append(flipped)
            junk_count += self.junk_card_from_bonus_card

            flipped = self._flip_card()

        return flipped, bonus_captures, junk_count

    def _flip_and_then_throw(
        self,
        card_thrown: Optional[Card],
        captures_before: CardList,
        junk_count_before: int,
    ) -> Tuple[bool, Card, CardList, Optional[CardList], int]:
        """
        Execute `self._flip_card_until_normal` and then throw the flipped card.

        `card_thrown` is the card thrown before flipping,
        `captures_before` is the list of cards captured before flipping, and
        `junk_count_before` is the number of junk cards that are taken from the opponent
        before flipping.

        `self._flip_and_then_throw` returns the following five variables:
        `flag_continue`: When a "double matches" is made by throwing the flipped card,
            we set this variable to False. Otherwise, it is set to True.
        `card_flipped`: The flipped card.
        `bonus_captures`: The list of bonus cards that were captured during flipping.
        `captures_after`: The list of cards that were captured by throwing the flipped card.
        `junk_count`: Total number of junk cards that will be brought from the opponent.
        """

        # flip the card on top of the drawing pile
        # until we get a normal [non-bonus] card
        (
            flipped,
            bonus_captures,
            junk_count_flip,
        ) = self._flip_card_until_normal()

        # throw the flipped normal card
        (captures_after, junk_count_after) = self._throw_card(flipped, card_thrown)

        # sum all junk counts
        junk_count = junk_count_before + junk_count_flip + junk_count_after

        # If the player needs to select a match,
        if captures_after is None:
            assert self.state.select_match is not None
            self.state.select_match = (
                self.state.select_match[0],
                self.state.select_match[1],
                (
                    CardList(captures_before),
                    flipped,
                    CardList(bonus_captures),
                    junk_count,
                ),
            )
            return (False, flipped, bonus_captures, captures_after, junk_count)

        return (True, flipped, bonus_captures, captures_after, junk_count)

    def _check_after_flip(
        self,
        card: Optional[Card],
        captures_before: CardList,
        flipped: Card,
        bonus_captures: CardList,
        captures_after: CardList,
    ) -> int:
        """
        Check for discard-and-match, ttadak, stacking, or clear; and tune the junk count.

        For instance, consider the following case:
            self._flip_card_until_normal() increased the `junk_count`
            and it ended with a stacking so that the increment of junk count should be cancelled.

        In this case, self._check_after_flip(...) will return
        `-self.junk_card_from_bonus_card` to reverse the increment of `junk_count`.

        It will return the amount of adjustment for the junk count.
        """

        board = self.board
        state = self.state

        junk_count = 0

        if (
            card is not None
            and flipped.month == card.month
            and isinstance(card.month, int)
        ):
            if captures_before == []:
                # discard-and-match except the last turn
                if board.hands[state.player] != []:
                    self.logger.log("discard and match", (card, flipped))
                    junk_count += self.junk_card_from_discard_and_match

                # captures_after should have exactly two elements,
                # `action.card` and `flipped`
                assert len(captures_after) == 2

                # capture all
                self._append_to_capture_field(CardList(bonus_captures + captures_after))

            elif len(captures_before) == 2 and captures_after == []:
                # stacking except the last turn
                if board.hands[state.player] != []:
                    self._append_to_center_field(
                        CardList(captures_before + bonus_captures)
                    )
                    last = board.center_field[card.month].pop(0)
                    board.center_field[card.month].append(last)
                    assert flipped.month is not None
                    state.stacking_histories[state.player].add(flipped.month)
                    self.logger.log("stacking", board.center_field[card.month])
                    junk_count -= self.junk_card_from_bonus_card * len(bonus_captures)

                    # three stackings
                    if len(state.stacking_histories[state.player]) >= 3:
                        state.ended = True
                        state.winner = state.player
                        state.score_factors[state.player] = [
                            ScoreFactor("three stackings")
                        ]
                        return 0

                else:
                    self._append_to_capture_field(
                        CardList(captures_before + bonus_captures)
                    )

            elif len(captures_before) == 2 and len(captures_after) == 2:
                # Check for "ttadak."
                self.logger.log("ttadak", card.month)
                self._append_to_capture_field(
                    CardList(captures_before + bonus_captures + captures_after)
                )
                assert flipped.month is not None
                state.stacking_histories[state.player].add(flipped.month)
                junk_count += self.junk_card_from_ttadak

        else:
            # Discard-and-match, ttadak, and stacking are not occurred.
            self._append_to_capture_field(
                CardList(captures_before + bonus_captures + captures_after)
            )

        # Check for "clear."
        if (
            all(field == [] for field in board.center_field.values())
            and board.hands[state.player] != []
        ):
            self.logger.log("clear")
            junk_count += self.junk_card_from_clear

        return junk_count

    def _take_junk_from_opponent(self, junk_count: int) -> None:
        """
        Take `junk_count` junk cards from the opponent.

        The number of junk cards will be the minimum of possible combination of junk cards,
        at least `junk_count`, if there are enough amount of junk cards.

        For instance, if `junk_count` is 1 but the opponent has no other junk (or bonus) cards
        than a triple bonus card in the capture field, then the triple bonus card are taken.

        For another example, if `junk_count` is 3 but the opponent has 2 normal junk cards,
        then all those cards are taken.
        """

        board = self.board
        state = self.state
        opponent = get_opponent(state.player)

        remaining_junk_count = junk_count

        while remaining_junk_count > 0:
            singles = [
                card
                for card in board.capture_fields[opponent]
                if card.kind == "junk" and card.multiple == 1
            ]
            doubles = [
                card
                for card in board.capture_fields[opponent]
                if card.kind in {"junk", "bonus"} and card.multiple == 2
            ] + (
                [AnimalCard(9)]
                if state.animal_9_moved
                and AnimalCard(9) in board.capture_fields[opponent]
                else []
            )
            triples = (
                [BonusCard(3)] if BonusCard(3) in board.capture_fields[opponent] else []
            )

            if len(singles) == 0 and len(doubles) == 0 and len(triples) == 0:
                return

            if len(triples) > 0 and remaining_junk_count >= 3:
                junk = triples[-1]
                remaining_junk_count -= 3
            elif len(doubles) > 0 and remaining_junk_count >= 2:
                junk = doubles[-1]
                remaining_junk_count -= 2
            elif len(singles) > 0:
                junk = singles[-1]
                remaining_junk_count -= 1
            elif len(doubles) > 0:
                junk = doubles[-1]
                remaining_junk_count -= 2
            elif len(triples) > 0:
                junk = triples[-1]
                remaining_junk_count -= 3

            self.board.capture_fields[opponent].remove(junk)
            self.board.capture_fields[state.player].append(junk)
            self.logger.log("take junk from opponent", junk)

    def calculate_scores(self, without_multiples: bool = False):
        """Set the list of score factors and calculate the scores."""

        kinds = [f.kind for f in self.state.score_factors[self.state.player]]
        if "four of a month" in kinds or "three stackings" in kinds:
            self.state.scores = [
                calculate_score(self.state, player) for player in [0, 1]
            ]
            return

        for player in [0, 1]:
            opponent = get_opponent(player)

            self.state.score_factors[player] = []

            capture_field = self.board.capture_fields[player]

            # 1. bright
            # three/four brights
            bright_point = len([card for card in capture_field if card.kind == "bright"])

            if bright_point < 3:
                bright_point = 0

            # three brights with rain (December)
            if bright_point == 3 and BrightCard(12) in capture_field:
                bright_point = 2

            # five brights
            if bright_point == 5:
                bright_point = 15

            self.state.score_factors[player].append(
                ScoreFactor("bright", bright_point)
            )

            # 2. animal
            num_of_animals = len(
                capture_field.apply_filter(lambda card: card.kind == "animal")
            )
            if AnimalCard(9) in capture_field and self.state.animal_9_moved:
                num_of_animals -= 1
            self.state.score_factors[player].append(
                ScoreFactor("animal", num_of_animals)
            )

            # 3. ribbon
            num_of_ribbons = len(
                capture_field.apply_filter(lambda card: card.kind == "ribbon")
            )
            self.state.score_factors[player].append(
                ScoreFactor("ribbon", num_of_ribbons)
            )

            # 4. junk
            junk_count = sum(
                card.multiple for card in capture_field if card.kind in {"junk", "bonus"}
            )
            if AnimalCard(9) in capture_field and self.state.animal_9_moved:
                junk_count += 2

            self.state.score_factors[player].append(
                ScoreFactor("junk", junk_count)
            )

            # 5. five birds
            if {AnimalCard(2), AnimalCard(4), AnimalCard(8)}.issubset(capture_field):
                self.state.score_factors[player].append(
                    ScoreFactor("five birds")
                )

            # 6. blue/red/plant ribbons
            if {RibbonCard(6), RibbonCard(9), RibbonCard(10)}.issubset(capture_field):
                self.state.score_factors[player].append(
                    ScoreFactor("blue ribbons")
                )

            if {RibbonCard(1), RibbonCard(2), RibbonCard(3)}.issubset(capture_field):
                self.state.score_factors[player].append(
                    ScoreFactor("red ribbons")
                )

            if {RibbonCard(4), RibbonCard(5), RibbonCard(7)}.issubset(capture_field):
                self.state.score_factors[player].append(
                    ScoreFactor("plant ribbons")
                )

            # 7. go
            self.state.score_factors[player].append(
                ScoreFactor("go", len(self.state.go_histories[player]))
            )

            if not without_multiples and player == self.state.winner:
                # 8. shaking
                self.state.score_factors[player].append(
                    ScoreFactor(
                        "shaking",
                        len(self.state.shaking_histories[player]),
                    )
                )
                # 9. penalties
                opponent_capture_field = self.board.capture_fields[opponent]
                # bright penalty
                if (
                    opponent_capture_field.apply_filter(lambda card: card.kind == "bright")
                    == []
                    and bright_point > 0
                ):
                    self.state.score_factors[player].append(
                        ScoreFactor("bright penalty")
                    )

                # animal penalty
                if num_of_animals >= 7:
                    self.state.score_factors[player].append(
                        ScoreFactor("animal penalty")
                    )

                # junk penalty
                opponent_junk_count = sum(
                    card.multiple
                    for card in opponent_capture_field
                    if card.kind in {"junk", "bonus"}
                )
                if AnimalCard(9) in opponent_capture_field and self.state.animal_9_moved:
                    opponent_junk_count += 2

                if 1 <= opponent_junk_count <= 7 and junk_count >= 10:
                    self.state.score_factors[player].append(
                        ScoreFactor("junk penalty")
                    )

                # go penalty
                if (
                    self.state.go_histories[opponent] != []
                    and self.state.winner == player
                ):
                    self.state.score_factors[player].append(
                        ScoreFactor("go penalty")
                    )

        # Refer to `calculate_score`.
        self.state.scores = [calculate_score(self.state, player) for player in [0, 1]]

    def _go(self) -> None:
        """The current player (`self.state.player`) declares Go."""

        state = self.state

        state.go_histories[state.player].append(0)
        self.calculate_scores(without_multiples=True)
        state.go_histories[state.player][-1] = self.state.scores[state.player]

        state.player = cast(Player, 1 - state.player)

        self.logger.log("go", True)

    def _stop(self) -> None:
        """The current player (`self.state.player`) declares Stop."""

        board = self.board
        state = self.state

        state.ended = True
        state.winner = state.player

        self.calculate_scores()
        score_before_moving_animal_9 = state.scores[state.player]

        # Move animal 9 if the opponent had not decided yet to move it or not
        # and it is better to do so for the opponent.
        if (
            state.animal_9_moved is None
            and AnimalCard(9) in board.capture_fields[1 - state.player]
        ):
            state.animal_9_moved = True

            self.calculate_scores()
            score_after_moving_animal_9 = state.scores[state.player]

            # when it gets worse by moving animal 9, roll it back
            if score_after_moving_animal_9 > score_before_moving_animal_9:
                state.animal_9_moved = False

            else:
                self.logger.log("move animal 9")

            self.calculate_scores()

        self.logger.log("go", False)

    def serialize(self) -> dict:
        """Serialize the game"""
        return {
            "board": self.board.serialize(),
            "state": self.state.serialize(),
            "flags": self.flags.serialize(),
            "logs": self.logger.serialize(),
        }

    @staticmethod
    def deserialize(data: dict):
        """Deserialize the game"""

        game = Game()
        game.board = Board.deserialize(data["board"])
        game.state = State.deserialize(data["state"])
        game.flags = Flags.deserialize(data["flags"])

        return game
