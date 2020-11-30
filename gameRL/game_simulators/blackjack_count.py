# Created by Patrick Kao
import math
from typing import Tuple

import numpy as np
from gym import spaces

from gameRL.game_simulators.blackjack import BlackjackDeck, CARD_VALUES, SUITS, BlackjackHand, \
    DEALER_MAX, BlackjackCustomEnv


class BlackjackDeckwithCount(BlackjackDeck):
    def __init__(self, N_decks: int, with_replacement=False, rho=1):
        BlackjackDeck.__init__(self, N_decks, with_replacement)
        self.count = 0
        self.rho = rho
        self.reshuffle_point = math.floor(
            len(CARD_VALUES) * SUITS * self.N_decks * (1 - self.rho)
        )
        self.cards_used = 0
        self.reshuffled = False

    def draw_card(self) -> Tuple[int, bool]:
        """Draws and returns card from the deck"""
        if self.reshuffled:
            return None, self.reshuffled
        if len(self.deck) - 1 <= self.reshuffle_point:
            self.reshuffled = True
        self.cards_used += 1
        index = np.random.randint(len(self.deck))
        self.count += self.update_count(self.deck[index])
        if self.with_replacement:
            return self.deck[index], self.reshuffled
        return self.deck.pop(index), self.reshuffled

    def update_count(self, card, system="Hi-Lo") -> int:
        """
        Computes various card-counting systems including: Hi-Lo
        """
        if system == "Hi-Lo":
            if card in [2, 3, 4, 5, 6]:
                return 1
            elif card in [7, 8, 9]:
                return 0
            else:
                return -1

    def get_running_count(self) -> int:
        return self.count

    def _get_num_decks(self) -> int:
        cards_per_deck = len(CARD_VALUES) * SUITS
        num_decks = math.ceil(len(self.deck) / cards_per_deck)
        return num_decks

    def _get_cards_used(self) -> int:
        return self.cards_used

    def _get_full_deck_size(self) -> int:
        return len(CARD_VALUES) * SUITS * self.N_decks


class BlackjackHandwithReshuffle(BlackjackHand):
    def __init__(self, blackjack_deck: BlackjackDeckwithCount, max_hand_sum: int = 21):
        BlackjackHand.__init__(self, blackjack_deck, max_hand_sum)
        self.reshuffled = False

    def draw_card(self):
        card, reshuffled = self.blackjack_deck.draw_card()
        if not reshuffled:
            self.hand.append(card)
        else:
            self.reshuffled = True


class BlackjackEnvwithRunningCount(BlackjackCustomEnv):
    def __init__(self, N_decks: int, natural_bonus: bool = True, rho=1, max_hand_sum: int = 21):
        BlackjackCustomEnv.__init__(self, N_decks, natural_bonus, max_hand_sum=max_hand_sum)
        # actions: either "hit" (keep playing), "stand" (stop where you are), observe or join
        self.action_space = spaces.Discrete(5)
        # count observation depends on the card-counting system and number of decks
        # use the following defaults
        # Hi-Lo: [-20 * N_decks, 20 * N_decks], (2*20 + 1) * N_decks
        count_space = (2 * 20 + 1) * N_decks
        self.observation_space = spaces.MultiDiscrete(
            [33, 11, 2, count_space, 2])  # last for observing or not

        self.blackjack_deck: BlackjackDeck = BlackjackDeckwithCount(
            self.N_decks, rho=rho
        )
        self.observing = True
        self.reshuffled = False

        self.reset()

    def _calculate_player_reward(self) -> int:
        """
        Computes the player's reward in the case that neither busts
        -1 for dealer > player, 0 for tie, 1 for player > dealer
        """
        if self.observing:
            return 0
        player_sum = self.player.score()
        dealer_sum = self.dealer.score()
        return (player_sum > dealer_sum) - (dealer_sum > player_sum)

    def _hit(self) -> Tuple[bool, int]:
        """Handles case where the player chooses to hit"""
        hand_done = False
        if self.observing:  # return early if player is observing
            hand_done = True
            return hand_done, 0

        self.player.draw_card()
        if self.player.reshuffled:  # Deck ran out of cards
            self.reshuffled = True
            hand_done = True
            reward = 0
        elif self.player.is_bust():
            hand_done = False
            reward = -1
        else:
            hand_done = False
            reward = 0
        return hand_done, reward

    def _stick(self) -> Tuple[bool, int]:
        """Handles case where the player chooses to stick"""
        hand_done = True
        if self.observing:  # return early if player is observing
            return hand_done, 0

        while self.dealer.sum_hand() < DEALER_MAX:
            self.dealer.draw_card()
            if self.dealer.reshuffled:  # Return early if run out of cards
                self.reshuffled = True
                return hand_done, 0

        reward = self._calculate_player_reward()
        if self.natural_bonus and self.player.is_natural() and reward == 1:
            reward = 1.5

        return hand_done, reward

    def _dummy_stick(self) -> Tuple[bool, int]:
        hand_done = True
        while self.dealer.sum_hand() < DEALER_MAX:
            self.dealer.draw_card()
            if self.dealer.reshuffled:  # Return early if run out of cards
                self.reshuffled = True
                return hand_done, 0

        reward = 0  # If the player is not in the game, they should receive no reward
        if (
                self.player
        ):  # if player is in a hand switches to observing, assume he sticks
            reward = self._calculate_player_reward()
            if self.natural_bonus and self.player.is_natural() and reward == 1:
                reward = 1.5

        return hand_done, reward

    def _double_down(self):
        """
        Handles case where the player chooses to double down
        If the double down is illegeal, just ignore it
        """
        # it is illegal to double down if you do not have a 9, 10
        hand_done = True
        if self.observing:  # return early if player is observing
            return hand_done, 0

        multiplier = 2
        if not self.player.is_double_down_legal():
            return False, 0

        done, reward = self._hit()
        # case where you went over
        if done:
            return done, multiplier * reward
        _, reward = self._stick()
        return True, multiplier * reward

    def step(self, action) -> Tuple[Tuple, int, bool, dict]:
        """Action must be in the set {0,1,2,3}"""
        assert self.action_space.contains(action)
        # player hits
        game_done = False
        if action == 1:
            hand_done, reward = self._hit()
        elif action == 0:  # player sticks
            hand_done, reward = self._stick()
        elif action == 2:  # player joins
            self.observing = False
            hand_done = True
            reward = 0
        elif action == 3:  # player observes:
            self.observing = True
            hand_done, reward = self._dummy_stick()
        else:  # player doubles down
            hand_done, reward = self._double_down()

        if hand_done:  # draw new cards
            self.redeal()

        if self.dealer.reshuffled or self.dummy.reshuffled:
            self.reshuffled = True

        game_done = game_done or self.reshuffled

        return self._get_obs(), reward, game_done, {}

    def _get_obs(self) -> Tuple[int, int, bool, int, bool]:
        """
        Gets player's current obs
        :return: Returns sum of own hand, dealer card, usable ace, card counting obs, observing flag
        """
        if self.reshuffled:
            return (
                0,
                1,
                False,
                self.blackjack_deck.get_running_count(),
                self.observing,
            )
        if self.observing:
            return (
                0,
                self.dealer.hand[0],
                False,
                self.blackjack_deck.get_running_count(),
                self.observing,
            )
        else:
            return (
                self.player.sum_hand(),
                self.dealer.hand[0],
                self.player.has_usable_ace(),
                self.blackjack_deck.get_running_count(),
                self.observing,
            )

    def reset(self) -> Tuple[int, int, bool]:
        if not hasattr(self, "blackjack_deck"):
            return None
        self.dealer = BlackjackHandwithReshuffle(self.blackjack_deck)
        self.dummy = BlackjackHandwithReshuffle(self.blackjack_deck)
        if not self.observing:
            self.player = BlackjackHandwithReshuffle(self.blackjack_deck)
        else:
            self.player = None
        return self._get_obs()

    def redeal(self) -> Tuple[int, int, bool]:
        self.dealer._initial_draw()
        self.dummy._initial_draw()
        self.reshuffled = (
                self.reshuffled or self.dealer.reshuffled or self.dummy.reshuffled
        )
        if not self.observing:
            if not self.player:
                self.player = BlackjackHandwithReshuffle(self.blackjack_deck)
            self.player._initial_draw()
            self.reshuffled = self.reshuffled or self.player.reshuffled
        else:
            self.player = None

        return self._get_obs()


class BlackjackEnvwithTrueCount(BlackjackEnvwithRunningCount):
    def __init__(self, N_decks: int, natural_bonus: bool = True, rho=1):
        BlackjackEnvwithRunningCount.__init__(self, N_decks, natural_bonus, rho=rho)
        # self.action_space = spaces.Discrete(4)
        true_min, true_max = -20, 20
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(33),  # 32 + 1 for observing hand sum of 0
                spaces.Discrete(11),
                spaces.Discrete(2),
                spaces.Box(
                    np.array([true_min], dtype=np.float32),
                    np.array([true_max], dtype=np.float32),
                ),
                spaces.Discrete(2),  # observing or not
            )
        )
        # self.blackjack_deck: BlackjackDeck = BlackjackDeckwithCount(
        #     self.N_decks, rho=rho
        # )
        # self.observing = True
        # self.reshuffled = False

        # self.reset()

    def _get_obs(self) -> Tuple[int, int, bool]:
        if self.reshuffled:
            return (
                0,
                1,
                False,
                self.blackjack_deck.get_true_count(),
                self.observing,
            )
        if self.observing:
            return (
                0,
                self.dealer.hand[0],
                False,
                self.blackjack_deck.get_true_count(),
                self.observing,
            )
        else:
            return (
                self.player.sum_hand(),
                self.dealer.hand[0],
                self.player.has_usable_ace(),
                self.blackjack_deck.get_true_count(),
                self.observing,
            )
