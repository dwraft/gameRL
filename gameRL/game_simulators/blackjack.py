"""
Modeled largely after
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
Also, the github version draws with replacement, while I modified to not use replacement

Also, reference here for how to play blackjack
"""
from typing import Dict, List, Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

SUITS = 4
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
MAX_HAND_SUM = 21
DEALER_MAX = 17


class BlackjackDeck:
    def __init__(self, N_decks: int, with_replacement=False):
        self.N_decks = N_decks
        self.deck = CARD_VALUES.copy() * SUITS * N_decks
        self.with_replacement = with_replacement
        self.count = 0

    def draw_card(self) -> Tuple[int, bool]:
        """Draws and returns card from the deck"""
        reshuffled = False
        if len(self.deck) == 0:
            self.deck = CARD_VALUES.copy() * SUITS * N_decks
            self.count = 0
            reshuffled = True
        index = np.random.randint(len(self.deck))
        self.count += self.update_count(self.deck[index])
        if self.with_replacement:
            return self.deck[index], reshuffled
        return self.deck.pop(index), reshuffled

    def update_count(self, card, system="Hi-Lo") -> Union[int, float]:
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

    def get_count(self) -> Union[int, float]:
        return self.count


class BlackjackHand:
    def __init__(self, blackjack_deck: BlackjackDeck):
        self.blackjack_deck: BlackjackDeck = blackjack_deck
        self.hand: List[int] = []
        self._initial_draw()
        self.reshuffled = False

    def draw_card(self):
        card, reshuffled = self.blackjack_deck.draw_card()
        if not reshuffled:
            self.hand.append(card)
        else:
            self.reshuffled = True

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
        self.action_space = spaces.Discrete(2)

        # count observation depends on the card-counting system and number of decks
        # use the following defaults
        # Hi-Lo: [-20 * N_decks, 20 * N_decks], (2*20 + 1) * N_decks
        count_space = (2 * 20 + 1) * N_decks
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(33),  # 32 + 1 for observing hand sum of 0
                spaces.Discrete(11),
                spaces.Discrete(2),
                spaces.Discrete(count_space),
                spaces.Discrete(2),  # observing or not
            )
        )

        self.N_decks = N_decks
        self.seed()

        self.observing = True

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
        if self.observing:
            return 0
        player_sum = self.player.score()
        dealer_sum = self.dealer.score()
        return (player_sum > dealer_sum) - (dealer_sum > player_sum)

    def _hit(self):
        """Handles case where the player chooses to hit"""
        if self.observing:  # penalize if player is observing
            return False, -1000

        self.player.draw_card()
        if self.player.is_bust():
            done = False
            reward = -1
        elif self.player.reshuffled:  # Deck ran out of cards
            done = True
            reward = 0
        else:
            done = False
            reward = 0
        return done, reward

    def _stick(self):
        """Handles case where the player chooses to stick"""
        if self.observing:  # penalize if player is observing
            return False, -1000

        done = False
        while self.dealer.sum_hand() < DEALER_MAX and not self.dealer.reshuffled:
            self.dealer.draw_card()

        if self.dealer.reshuffled:  # Return early if run out of cards
            return True, 0

        reward = self._calculate_player_reward()
        if self.natural_bonus and self.player.is_natural() and reward == 1:
            reward = 1.5

        return done, reward

    def _dummy_stick(self):
        done = False
        while self.dealer.sum_hand() < DEALER_MAX and not self.dealer.reshuffled:
            self.dealer.draw_card()

        if self.dealer.reshuffled:  # Return early if run out of cards
            done = True

        return done, 0

    def _get_info(self) -> Dict:
        """Return debugging info, for now just empty dictionary"""
        return {}

    def step(self, action):
        """Action must be in the set {0,1,2,3}"""
        assert self.action_space.contains(action)
        # player hits
        if action == 1:
            done, reward = self._hit()
        elif action == 0:  # player sticks
            done, reward = self._stick()
        elif action == 2:  # player joins
            self.observing = False
            self.player = BlackjackHand(self.blackjack_deck)
            done = self.player.reshuffled
            reward = 0
        else:  # player observes
            self.observing = True
            done, reward = self._dummy_stick()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.observing:
            return (
                0,
                self.dealer.hand[0],
                False,
                self.blackjack_deck.get_count(),
                self.observing,
            )
        else:
            return (
                self.player.sum_hand(),
                self.dealer.hand[0],
                self.player.has_usable_ace(),
                self.blackjack_deck.get_count(),
                self.observing,
            )

    def reset(self):
        self.blackjack_deck: BlackjackDeck = BlackjackDeck(self.N_decks)
        self.dealer = BlackjackHand(self.blackjack_deck)
        self.dummy = BlackjackHand(self.blackjack_deck)
        # self.player = BlackjackHand(self.blackjack_deck)
        return self._get_obs()
