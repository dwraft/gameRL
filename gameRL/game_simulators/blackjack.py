"""
Modeled largely after
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
Also, the github version draws with replacement, while I modified to not use replacement

Also, reference here for how to play blackjack
"""
from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

SUITS = 4
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
MAX_HAND_SUM = 21
DEALER_MAX = 17

ALLOWED_DOUBLE_DOWN_SUMS = {0,10,11}


class BlackjackDeck:
    def __init__(self, N_decks: int, with_replacement=False):
        self.N_decks = N_decks
        self.deck = CARD_VALUES.copy() * SUITS * N_decks
        self.with_replacement = with_replacement

    def draw_card(self) -> int:
        """Draws and returns card from the deck"""
        index = np.random.randint(len(self.deck))
        if self.with_replacement:
            return self.deck[index]
        return self.deck.pop(index)

    def is_empty(self) -> bool:
        return not len(self.deck)


class BlackjackHand:
    def __init__(self, blackjack_deck: BlackjackDeck):
        self.blackjack_deck: BlackjackDeck = blackjack_deck
        self.hand: List[int] = []
        self._initial_draw()

    def draw_card(self):
        self.hand.append(self.blackjack_deck.draw_card())

    @property
    def hand_size(self) -> int:
        return len(self.hand)

    def is_double_down_legal(self) -> bool:
        return self.hand_size == 2 and self.score() in ALLOWED_DOUBLE_DOWN_SUMS

    def _initial_draw(self):
        for _ in range(2):
            self.draw_card()

    def has_usable_ace(self) -> bool:
        return 1 in self.hand and sum(self.hand) + 10 <= MAX_HAND_SUM

    def sum_hand(self) -> int:
        if self.has_usable_ace():
            return sum(self.hand) + 10
        return sum(self.hand)

    def is_bust(self) -> bool:
        return sum(self.hand) > MAX_HAND_SUM

    def score(self) -> int:
        return 0 if self.is_bust() else self.sum_hand()

    def is_natural(self) -> bool:
        """The optimal blackjack hand, eq"""
        return sorted(self.hand) == [1, 10]

    def __str__(self):
        return f"Hand={self.hand}  Score={self.score()}"

    def __repr__(self):
        return f"Hand={self.hand}  Score={self.score()}"


class BlackjackCustomEnv(gym.Env):
    def __init__(self, N_decks, natural_bonus=True):
        # actions: either "hit" (keep playing) or "stand" (stop where you are)
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.MultiDiscrete([32, 11, 2])

        self.N_decks = N_decks
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural_bonus = natural_bonus
        # start the first game
        self.reset()

    def render(self) -> None:
        print(f"Dealer State: {str(self.dealer)}\n Player State: {str(self.player)}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_player_reward(self) -> int:
        """
        Computes the player's reward in the case that neither busts
        -1 for dealer > player, 0 for tie, 1 for player > dealer
        """
        player_sum = self.player.score()
        dealer_sum = self.dealer.score()
        return (player_sum > dealer_sum) - (dealer_sum > player_sum)

    def _hit(self):
        """Handles case where the player chooses to hit"""
        self.player.draw_card()
        if self.player.is_bust():
            done = True
            reward = -1
        else:
            done = False
            reward = 0
        return done, reward

    def _stick(self):
        """Handles case where the player chooses to stick"""
        done = True
        while self.dealer.sum_hand() < DEALER_MAX:
            self.dealer.draw_card()
        reward = self._calculate_player_reward()
        if self.natural_bonus and self.player.is_natural() and reward == 1:
            reward = 1.5

        return done, reward

    def _double_down(self):
        """
        Handles case where the player chooses to double down
        If the double down is illegeal, just ignore it
        """
        # it is illegal to double down if you do not have a 9, 10

        multiplier = 2
        if not self.player.is_double_down_legal():
            return False, 0
        done, reward = self._hit()
        # case where you went over
        if done:
            return done, multiplier * reward
        _, reward = self._stick()
        return True, multiplier * reward

    def _get_info(self) -> Dict:
        """Return debugging info, for now just empty dictionary"""
        return {}

    def step(self, action):
        """Action must be in the set {0,1}"""
        assert self.action_space.contains(action)
        # player hits

        if action == 0:
            done, reward = self._stick()
        if action == 1:
            done, reward = self._hit()
        if action == 2:
            done, reward = self._double_down()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (
            self.player.sum_hand(),
            self.dealer.hand[0],
            self.player.has_usable_ace(),
        )

    def reset(self):
        self.blackjack_deck: BlackjackDeck = BlackjackDeck(self.N_decks)
        self.dealer = BlackjackHand(self.blackjack_deck)
        self.player = BlackjackHand(self.blackjack_deck)
        return self._get_obs()
