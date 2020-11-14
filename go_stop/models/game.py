import random
from copy import copy, deepcopy
from itertools import groupby
import json
from typing import List, Union

from .card import (
    Card,
    BrightCard,
    AnimalCard,
    RibbonCard,
    BonusCard,
    SpecialCard,
)
from .card_list import CardList
from ..constants.card import go_stop_cards

from .setting import Setting
from .board import Board
from .state import State
from .flags import Flags
from .logger import Logger
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
from .scorer import Scorer
from .score_factor import ScoreFactor


class Game(Setting):
    def __init__(self, player: int = 0):
        super().__init__()

        self.board = Board(player)
        self.state = State(player)
        self.flags = Flags()
        self.logger = Logger()

        four_of_a_month = self.board.four_of_a_month()

        if self.proceed_when_center_field_has_a_four_of_a_month:
            # if all players have a four-of-a-month
            # or the center field has a four-of-a-month,
            # reset the board
            while len(
                [
                    player
                    for player in {0, 1}
                    if len(four_of_a_month[player]) > 0
                ]
            ) == 2 or any(
                len(l) == 4 for l in self.board.center_field.values()
            ):
                self.board = Board()
                four_of_a_month = self.board.four_of_a_month()
        else:
            # if all players have a four-of-a-month,
            # reset the board
            while (
                len(
                    [
                        player
                        for player in {0, 1}
                        if len(four_of_a_month[player]) > 0
                    ]
                )
                == 2
            ):
                self.board = Board()
                four_of_a_month = self.board.four_of_a_month()

        p = [player for player in {0, 1} if len(four_of_a_month[player]) > 0]
        if len(p) == 1:
            self.state.player = p[0]
            self.flags.four_of_a_month = True

    def actions(self) -> List[Action]:
        """
        Returns a list of possible actions with this game
        """

        board = self.board
        state = self.state
        flags = self.flags

        if state.ended:
            return []

        hand = board.hands[state.player]

        if flags.four_of_a_month:
            return [ActionFourOfAMonth(option) for option in {True, False}]

        if flags.go:
            return [ActionGo(option) for option in {True, False}]

        if flags.select_match:
            assert state.select_match != None
            _, matches, _ = state.select_match
            return [ActionSelectMatch(match) for match in matches]

        if flags.shaking:
            return [ActionShaking(option) for option in {True, False}]

        if flags.move_animal_9:
            return [ActionMoveAnimal9(option) for option in {True, False}]

        # normal play

        # shaking
        months_in_hand = [card.month for card in hand if card.month != None]
        months_in_hand.sort()
        center_field_months = [
            month for month in range(1, 13) if board.center_field[month] != []
        ]

        shakable_months = [
            month
            for month, group in groupby(months_in_hand)
            if len(list(group)) >= 3 and month not in center_field_months
        ]

        # bomb
        combined_hand = hand + [
            item for sublist in board.center_field.values() for item in sublist
        ]
        months_in_combined_hand = [
            card.month for card in combined_hand if card.month != None
        ]
        months_in_combined_hand.sort()

        bomb_months = [
            month
            for month, group in groupby(months_in_combined_hand)
            if len(list(group)) == 4 and len(hand.of_month(month)) >= 2
        ]

        return (
            [
                ActionThrow(card)
                for card in hand
                if card.month not in shakable_months + bomb_months
                and card.kind != "bomb"
            ]
            + (
                [ActionThrowBomb()]
                if any(card.kind == "bomb" for card in hand)
                else []
            )
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
        """

        if not pass_check and action not in self.actions():
            return False

        board = self.board
        state = self.state
        flags = self.flags

        if action.kind in {"throw", "throw bomb", "bomb", "select match"}:
            # handles events on "throw" until flipping
            if action.kind == "throw" or action.kind == "throw bomb":
                if action.kind == "throw bomb":
                    action = ActionThrow(
                        next(
                            (
                                c
                                for c in board.hands[state.player]
                                if c.kind == "bomb"
                            ),
                            None,
                        )
                    )

                board.hands[state.player].remove(action.card)
                self.logger.log("throw", action.card)

                # if the player threw a bonus card
                if action.card.kind == "bonus":
                    # append it to the capture field
                    self._append_to_capture_field([action.card])

                    # take junk from the opponent
                    self._take_junk_from_opponent(
                        self.junk_card_from_bonus_card
                    )

                    # append the flipped card to the hand
                    flipped = self._flip_card()
                    self._append_to_hand([flipped])
                    self._calculate_scores(without_multiples=True)

                    return True

                # elsewise, throw a card (or a bomb)
                (captures_before, junk_count_before) = self._throw_card(
                    action.card, "before"
                )

                # if it signals to terminate `self.play`, terminate it.
                if captures_before == None:
                    return True

                (
                    continues,
                    flipped,
                    bonus_captures,
                    captures_after,
                    junk_count,
                ) = self._flip_and_throw_after(
                    action, captures_before, junk_count_before
                )
                if not continues:
                    return True

            # handles events on "bomb" until flipping
            elif action.kind == "bomb":
                month = action.arg
                self.logger.log("bomb", month)

                # if the player do a bomb, take one junk card from the opponent
                junk_count_before = self.junk_card_from_bomb

                # and captures all four cards
                captures_before = CardList(
                    [
                        card
                        for card in board.hands[state.player]
                        if card.month == month
                    ]
                    + board.center_field[month]
                )

                # replace two/three cards of a month into bomb cards
                bomb_number = len(board.hands[state.player].of_month(month)) - 1
                bombed_cards = board.hands[state.player].of_month(month)
                board.hands[state.player] = board.hands[
                    state.player
                ].except_month(month)

                for _ in range(bomb_number):
                    board.hands[state.player].append(
                        SpecialCard("bomb", state.bomb_increment)
                    )
                    state.bomb_increment += 1

                # empty the center field of corresponding month
                board.center_field[month] = []

                # append to shaking history
                state.shaking_histories[state.player].append(bombed_cards)

                (
                    continues,
                    flipped,
                    bonus_captures,
                    captures_after,
                    junk_count,
                ) = self._flip_and_throw_after(
                    action, captures_before, junk_count_before
                )
                if not continues:
                    return True

            elif action.kind == "select match":
                card, matches, arg = state.select_match
                assert action.match in matches

                if arg == None:
                    # before flip

                    captures_before = [card, action.match]
                    self.logger.log("select match", (card, action.match))
                    board.center_field[card.month].remove(action.match)

                    state.select_match = None
                    flags.select_match = False

                    (
                        continues,
                        flipped,
                        bonus_captures,
                        captures_after,
                        junk_count,
                    ) = self._flip_and_throw_after(action, captures_before, 0)
                    if not continues:
                        return True

                if arg != None:
                    # after flip
                    (
                        card,
                        captures_before,
                        flipped,
                        bonus_captures,
                        junk_count,
                    ) = arg
                    captures_after = [flipped, action.match]
                    self.logger.log("select match", (flipped, action.match))
                    board.center_field[flipped.month].remove(action.match)

                    state.select_match = None
                    flags.select_match = False

            # check discard-and-match, ttadak, stacking, or clear;
            # and tune the junk count.
            #
            # for instance, consider the following case:
            #     self._flip_card_until_normal() might increase
            #     `junk_count_flip` and it ends with a stacking so that
            #     that increment of junk count should be reversed.
            # in this case, self._check_after_flip(...) will return
            # `-self.junk_card_from_bonus_card` to reverse it.
            junk_count += self._check_after_flip(
                action.card if action.kind == "throw" else None,
                captures_before,
                flipped,
                bonus_captures,
                captures_after,
            )

            if state.ended:
                return True

            # take junks
            self._take_junk_from_opponent(junk_count)

            # calculate the score factors
            self._calculate_scores(without_multiples=True)

            # check move_animal_9
            if (
                AnimalCard(9) in board.capture_fields[state.player]
                and state.animal_9_moved == None
            ):
                # first, check the score of the player
                state.animal_9_moved = True
                self._calculate_scores(without_multiples=True)
                score_after_animal_9 = state.scores[state.player]

                state.animal_9_moved = None
                self._calculate_scores(without_multiples=True)

                if score_after_animal_9 >= 7:
                    flags.move_animal_9 = True
                    return True

            # check go
            if (
                state.go_histories[state.player] == []
                and Scorer.calculate(state, state.player) >= 7
            ) or (
                state.go_histories[state.player] != []
                and Scorer.calculate(state, state.player)
                > state.go_histories[state.player][-1]
            ):
                if board.hands[state.player] == []:
                    self._stop()
                    return True

                else:
                    flags.go = True
                    return True

            # pass the turn
            self.logger.log("turn change", (state.player, 1 - state.player))
            state.player = 1 - state.player

            # push back?
            if state.player == state.starting_player and board.hands[state.starting_player] == []:
                state.ended = True
                state.winner = None

            return True

        if action.kind == "shaking":
            card, cards = state.shaking

            flags.shaking = False
            state.shaking = None

            if action.option:
                self.logger.log("shaking", CardList(cards))
                state.shaking_histories[state.player].append(CardList(cards))

            return self.play(ActionThrow(card), pass_check=True)

        if action.kind == "shakable":
            flags.shaking = True
            state.shaking = (
                action.card,
                board.hands[state.player].of_month(action.card.month),
            )

            return True

        if action.kind == "four of a month":
            flags.four_of_a_month = False

            if action.option:
                state.ended = True
                state.winner = state.player
                state.score_factors[state.player] = [
                    ScoreFactor("four of a month")
                ]

                return True

            if len(board.hands[0]) == 10:
                state.player = 0

            return True

        if action.kind == "go":
            flags.go = False

            if action.option:
                # go
                self._go()
                return True

            else:
                # stop
                self._stop()
                return True

        if action.kind == "move animal 9":
            state.animal_9_moved = action.option
            flags.move_animal_9 = False
            self.logger.log("move animal 9", action.option)
            self._calculate_scores(without_multiples=True)

            # check go
            if (
                state.go_histories[state.player] == []
                and self.state.scores[state.player] >= 7
            ) or (
                state.go_histories[state.player] != []
                and self.state.scores[state.player]
                > state.go_histories[state.player][-1]
            ):
                if board.hands[state.player] == []:
                    self._stop()
                    return True

                else:
                    flags.go = True
                    return True

            # pass the turn
            self.logger.log("turn change", (state.player, 1 - state.player))
            state.player = 1 - state.player

            # push back?
            if state.player == state.starting_player and board.hands[state.starting_player] == []:
                state.ended = True
                state.winner = None

            return True

    def _append_to_hand(self, cards: CardList) -> None:
        """
        Append cards to the player's hand.
        """

        self.board.hands[self.state.player].extend(cards)
        self.logger.log("append to hand", (self.state.player, CardList(cards)))
        self._sort_board()

    def _append_to_center_field(self, cards: CardList) -> None:
        """
        Discard cards onto the center field.
        Here, cards should have the same months, except for bonus cards.
        """

        if cards == []:
            return

        # cards should have the same month except bonus cards
        assert len(set(card.month for card in cards if card.month != None)) == 1
        month = cards[0].month

        self.board.center_field[month].extend(cards)
        self.logger.log("append to center field", CardList(cards))
        self._sort_board()

    def _append_to_capture_field(self, cards: CardList) -> None:
        """
        Append cards to the player's capture field.
        """

        self.board.capture_fields[self.state.player].extend(cards)
        self.logger.log(
            "append to capture field", (self.state.player, CardList(cards))
        )

    def _sort_board(self) -> None:
        """
        Sorts the hands and the center field of the board.
        """

        self.board.sort()

    def _throw_card(
        self, card: Card, before_or_after: str
    ) -> (Union[CardList, None], int):
        """
        Throw a card (or a bomb), and returns
        (captures: CardList | None, junk_count: int).

        If captures == None, it means
        `self.play(action)` should be terminated (with some flags set).

        junk_count means the number of junks taken from the opponent.
        """

        board = self.board
        state = self.state
        flags = self.flags

        captures = CardList()
        junk_count = 0

        if card.kind == "bomb":
            return (captures, 0)

        assert card.month != None

        center_field_of_month = board.center_field[card.month]
        if center_field_of_month == []:
            self._append_to_center_field(CardList([card]))

        elif len(center_field_of_month) == 1:
            captures.extend([card, center_field_of_month[0]])
            self.logger.log("single match", (card, center_field_of_month[0]))
            board.center_field[card.month].remove(center_field_of_month[0])

        elif len(center_field_of_month) == 2:
            # check if two matches are all junk cards of same multiple:
            if (
                center_field_of_month[0].kind
                == center_field_of_month[1].kind
                == "junk"
                and center_field_of_month[0].multiple
                == center_field_of_month[1].multiple
            ):
                captures.extend([card, center_field_of_month[0]])
                self.logger.log("double matches", (card, center_field_of_month))
                board.center_field[card.month].remove(center_field_of_month[0])

            # else, set the select match flag
            else:
                flags.select_match = True
                state.select_match = [
                    card,
                    center_field_of_month,
                    None,
                ]
                self.logger.log("double matches", (card, center_field_of_month))
                return (None, 0)

        elif len(center_field_of_month) >= 3:
            junk_count += (
                self.junk_card_from_self_stacking
                if card.month in state.stacking_histories[state.player]
                else self.junk_card_from_stacking
            )
            junk_count += self.junk_card_from_bonus_card * (
                len(board.center_field[card.month]) - 3
            )
            self.logger.log("triple matches", (card, center_field_of_month))
            captures.extend([card] + center_field_of_month)

            board.center_field[card.month] = CardList()

        else:
            raise "Something wrong!"

        return (captures, junk_count)

    def _flip_card(self) -> Card:
        card = self.board.drawing_pile.pop(0)
        self.logger.log("flip", card)
        return card

    def _flip_card_until_normal(self) -> (Card, CardList, int):
        flipped = self._flip_card()
        bonus_captures = CardList()
        junk_count = 0

        while flipped.kind == "bonus":
            bonus_captures.append(flipped)
            junk_count += self.junk_card_from_bonus_card

            flipped = self._flip_card()

        return flipped, bonus_captures, junk_count

    def _flip_and_throw_after(
        self,
        action: Action,
        captures_before: CardList,
        junk_count_before: int,
    ) -> (bool, Card, CardList, CardList, int):
        # action, captures_before, junk_count_before -> continues, flipped, bonus_captures, captures_after, junk_count

        # flip the card on top of the drawing pile
        # until we get a normal [non-bonus] card
        (
            flipped,
            bonus_captures,
            junk_count_flip,
        ) = self._flip_card_until_normal()

        # throw the flipped normal card
        (captures_after, junk_count_after) = self._throw_card(flipped, "after")

        # sum all junk counts
        junk_count = junk_count_before + junk_count_flip + junk_count_after

        # if it signals to terminate `self.play`, terminate it.
        if captures_after == None:
            self.state.select_match[2] = (
                action.card if action.kind == "throw" else None,
                CardList(captures_before),
                flipped,
                CardList(bonus_captures),
                junk_count,
            )
            return (False, flipped, bonus_captures, captures_after, junk_count)

        return (True, flipped, bonus_captures, captures_after, junk_count)

    def _check_after_flip(
        self,
        card: Card,
        captures_before: CardList,
        flipped: Card,
        bonus_captures: CardList,
        captures_after: CardList,
    ) -> int:
        board = self.board
        state = self.state

        junk_count = 0

        if (
            card != None
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
                self._append_to_capture_field(bonus_captures + captures_after)

            elif len(captures_before) == 2 and captures_after == []:
                # stacking except the last turn
                if board.hands[state.player] != []:
                    self._append_to_center_field(
                        captures_before + bonus_captures
                    )
                    last = board.center_field[card.month].pop(0)
                    board.center_field[card.month].append(last)
                    state.stacking_histories[state.player].add(flipped.month)
                    self.logger.log("stacking", board.center_field[card.month])
                    junk_count -= self.junk_card_from_bonus_card * len(
                        bonus_captures
                    )

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
                        captures_before + bonus_captures
                    )

            elif len(captures_before) == 2 and len(captures_after) == 2:
                # ttadak
                self.logger.log("ttadak", card.month)
                self._append_to_capture_field(
                    captures_before + bonus_captures + captures_after
                )
                state.stacking_histories[state.player].add(flipped.month)
                junk_count += self.junk_card_from_ttadak

        else:
            self._append_to_capture_field(
                captures_before + bonus_captures + captures_after
            )

        if (
            all(field == [] for field in board.center_field.values())
            and board.hands[state.player] != []
        ):
            self.logger.log("clear")
            junk_count += self.junk_card_from_clear

        return junk_count

    def _take_junk_from_opponent(self, n: int) -> None:
        board = self.board
        state = self.state
        opponent = 1 - state.player

        i = n
        while i > 0:
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
                if state.animal_9_moved == True
                and AnimalCard(9) in board.capture_fields[opponent]
                else []
            )
            triples = (
                [BonusCard(3)]
                if BonusCard(3) in board.capture_fields[opponent]
                else []
            )

            if len(singles) == 0 and len(doubles) == 0 and len(triples) == 0:
                return

            if len(triples) > 0 and i >= 3:
                junk = triples[-1]
                i -= 3
            elif len(doubles) > 0 and i >= 2:
                junk = doubles[-1]
                i -= 2
            elif len(singles) > 0:
                junk = singles[-1]
                i -= 1
            elif len(doubles) > 0:
                junk = doubles[-1]
                i -= 2
            elif len(triples) > 0:
                junk = triples[-1]
                i -= 3

            self.board.capture_fields[opponent].remove(junk)
            self.board.capture_fields[state.player].append(junk)
            self.logger.log("take junk from opponent", junk)

    def _calculate_scores(self, without_multiples: bool = False):
        kinds = [
            f.kind for f in self.state.score_factors[self.state.player]
        ]
        if "four of a month" in kinds or "three stackings" in kinds:
            self.state.scores = [
                Scorer.calculate(self.state, player) for player in [0, 1]
            ]
            return

        self.state.score_factors[self.state.player] = []

        cf = self.board.capture_fields[self.state.player]

        # 1. bright
        # three/four brights
        bright_point = len([card for card in cf if card.kind == "bright"])

        if bright_point < 3:
            bright_point = 0

        # three brights with rain (December)
        if bright_point == 3 and BrightCard(12) in cf:
            bright_point = 2

        # five brights
        if bright_point == 5:
            bright_point == 15

        self.state.score_factors[self.state.player].append(
            ScoreFactor("bright", bright_point)
        )

        # 2. animal
        num_of_animals = len(
            cf.apply_filter(lambda card: card.kind == "animal")
        )
        if AnimalCard(9) in cf and self.state.animal_9_moved == True:
            num_of_animals -= 1
        self.state.score_factors[self.state.player].append(
            ScoreFactor("animal", num_of_animals)
        )

        # 3. ribbon
        num_of_ribbons = len(
            cf.apply_filter(lambda card: card.kind == "ribbon")
        )
        self.state.score_factors[self.state.player].append(
            ScoreFactor("ribbon", num_of_ribbons)
        )

        # 4. junk
        junk_count = sum(
            card.multiple for card in cf if card.kind in {"junk", "bonus"}
        )
        if AnimalCard(9) in cf and self.state.animal_9_moved == True:
            junk_count += 2

        self.state.score_factors[self.state.player].append(
            ScoreFactor("junk", junk_count)
        )

        # 5. five birds
        if {AnimalCard(2), AnimalCard(4), AnimalCard(8)}.issubset(cf):
            self.state.score_factors[self.state.player].append(
                ScoreFactor("five birds")
            )

        # 6. blue/red/plant ribbons
        if {RibbonCard(6), RibbonCard(9), RibbonCard(10)}.issubset(cf):
            self.state.score_factors[self.state.player].append(
                ScoreFactor("blue ribbons")
            )

        if {RibbonCard(1), RibbonCard(2), RibbonCard(3)}.issubset(cf):
            self.state.score_factors[self.state.player].append(
                ScoreFactor("red ribbons")
            )

        if {RibbonCard(4), RibbonCard(5), RibbonCard(7)}.issubset(cf):
            self.state.score_factors[self.state.player].append(
                ScoreFactor("plant ribbons")
            )

        # 7. go
        self.state.score_factors[self.state.player].append(
            ScoreFactor("go", len(self.state.go_histories[self.state.player]))
        )

        if not without_multiples:
            # 8. shaking
            self.state.score_factors[self.state.player].append(
                ScoreFactor(
                    "shaking",
                    len(self.state.shaking_histories[self.state.player]),
                )
            )
            # 9. penalties
            opposite_cf = self.board.capture_fields[1 - self.state.player]
            # bright penalty
            if (
                opposite_cf.apply_filter(lambda card: card.kind == "bright")
                == []
                and bright_point > 0
            ):
                self.state.score_factors[self.state.player].append(
                    ScoreFactor("bright penalty")
                )

            # animal penalty
            if num_of_animals >= 7:
                self.state.score_factors[self.state.player].append(
                    ScoreFactor("animal penalty")
                )

            # junk penalty
            opposite_junk_count = sum(
                card.multiple
                for card in opposite_cf
                if card.kind in {"junk", "bonus"}
            )
            if (
                AnimalCard(9) in opposite_cf
                and self.state.animal_9_moved == True
            ):
                opposite_junk_count += 2

            if 1 <= opposite_junk_count <= 7 and junk_count >= 10:
                self.state.score_factors[self.state.player].append(
                    ScoreFactor("junk penalty")
                )

            # go penalty
            if (
                self.state.go_histories[1 - self.state.player] != []
                and self.state.winner == self.state.player
            ):
                self.state.score_factors[self.state.player].append(
                    ScoreFactor("go penalty")
                )

        self.state.scores = [
            Scorer.calculate(self.state, player) for player in [0, 1]
        ]

    def _go(self) -> None:
        state = self.state

        state.go_histories[state.player].append(0)
        self._calculate_scores(without_multiples=True)
        state.go_histories[state.player][-1] = self.state.scores[state.player]

        state.player = 1 - state.player

        self.logger.log("go", True)

    def _stop(self) -> None:
        board = self.board
        state = self.state

        state.ended = True
        state.winner = state.player

        self._calculate_scores()
        score_before_moving_animal_9 = state.scores[state.player]

        # move animal 9 if it is not queried for the opponent
        if (
            state.animal_9_moved == None
            and AnimalCard(9) in board.capture_fields[1 - state.player]
        ):
            state.animal_9_moved = True

            self._calculate_scores()
            score_after_moving_animal_9 = state.scores[state.player]

            # when it gets worse by moving animal 9, roll it back
            if score_after_moving_animal_9 > score_before_moving_animal_9:
                state.animal_9_moved = False

            else:
                self.logger.log("move animal 9")

            self._calculate_scores()

        self.logger.log("go", False)

    def serialize(self) -> dict:
        return {
            "board": self.board.serialize(),
            "state": self.state.serialize(),
            "flags": self.flags.serialize(),
            "logs": self.logger.serialize(),
        }

    @staticmethod
    def deserialize(data: dict):
        game = Game()

        game.board = Board.deserialize(data["board"])
        game.state = State.deserialize(data["state"])
        game.flags = Flags.deserialize(data["flags"])

        return game