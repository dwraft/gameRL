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

    def draw_card(self) -> int:
        """Draws and returns card from the deck"""
        index = np.random.randint(len(self.deck))
        if self.with_replacement:
            return self.deck[index]
        return self.deck.pop(index)

    def is_empty(self) -> bool:
        return not len(self.deck)


class BlackjackDeckwithCount(BlackjackDeck):
    def __init__(self, N_decks: int, with_replacement=False):
        BlackjackDeck.__init__(self, N_decks, with_replacement)
        self.count = 0

    def draw_card(self) -> Tuple[int, bool]:
        """Draws and returns card from the deck"""
        reshuffled = False
        if len(self.deck) == 0:
            self.deck = CARD_VALUES.copy() * SUITS * self.N_decks
            self.count = 0
            reshuffled = True
        index = np.random.randint(len(self.deck))
        self.count += self.update_count(self.deck[index])
        if self.with_replacement:
            return self.deck[index], reshuffled
        return self.deck.pop(index), reshuffled

    def update_count(self, card, system="Hi-Lo") -> Tuple[int, float]:
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

    def get_count(self) -> Tuple[int, float]:
        return self.count


class BlackjackHand:
    def __init__(self, blackjack_deck: BlackjackDeck):
        self.blackjack_deck: BlackjackDeck = blackjack_deck
        self.hand: List[int] = []
        self._initial_draw()

    def draw_card(self):
        self.hand.append(self.blackjack_deck.draw_card())

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

    def __str__(self) -> str:
        return f"Hand={self.hand}  Score={self.score()}"

    def __repr__(self) -> str:
        return f"Hand={self.hand}  Score={self.score()}"


class BlackjackHandwithReshuffle(BlackjackHand):
    def __init__(self, blackjack_deck: BlackjackDeckwithCount):
        BlackjackHand.__init__(self, blackjack_deck)
        self.reshuffled = False

    def draw_card(self):
        card, reshuffled = self.blackjack_deck.draw_card()
        if not reshuffled:
            self.hand.append(card)
        else:
            self.reshuffled = True


class BlackjackCustomEnv(gym.Env):
    def __init__(self, N_decks: int, natural_bonus: bool = True):
        # actions: either "hit" (keep playing) or "stand" (stop where you are)
        self.action_space = spaces.Discrete(2)

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

    def _hit(self) -> Tuple[bool, int]:
        """Handles case where the player chooses to hit"""
        self.player.draw_card()
        if self.player.is_bust():
            done = True
            reward = -1
        else:
            done = False
            reward = 0
        return done, reward

    def _stick(self) -> Tuple[bool, int]:
        """Handles case where the player chooses to stick"""
        done = True
        while self.dealer.sum_hand() < DEALER_MAX:
            self.dealer.draw_card()
        reward = self._calculate_player_reward()
        if self.natural_bonus and self.player.is_natural() and reward == 1:
            reward = 1.5

        return done, reward

    def _get_info(self) -> Dict:
        """Return debugging info, for now just empty dictionary"""
        return {}

    def step(self, action) -> Tuple[Tuple, int, bool, dict]:
        """Action must be in the set {0,1}"""
        assert self.action_space.contains(action)
        # player hits
        if action == 1:
            done, reward = self._hit()
        else:
            done, reward = self._stick()
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> Tuple[int, int, bool]:
        return (
            self.player.sum_hand(),
            self.dealer.hand[0],
            self.player.has_usable_ace(),
        )

    def reset(self) -> Tuple[int, int, bool]:
        self.blackjack_deck: BlackjackDeck = BlackjackDeck(self.N_decks)
        self.dealer = BlackjackHand(self.blackjack_deck)
        self.player = BlackjackHand(self.blackjack_deck)
        return self._get_obs()


class BlackjackEnvwithCount(BlackjackCustomEnv):
    def __init__(self, N_decks: int, natural_bonus: bool = True):
        BlackjackCustomEnv.__init__(self, N_decks, natural_bonus)
        # actions: either "hit" (keep playing), "stand" (stop where you are), observe or join
        self.action_space = spaces.Discrete(4)
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

        self.blackjack_deck: BlackjackDeck = BlackjackDeckwithCount(self.N_decks)
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
        else:  # player observes
            self.observing = True
            hand_done, reward = self._dummy_stick()

        if hand_done:  # draw new cards
            self.reset()  # draw dealer and dummy

        if self.dealer.reshuffled or self.dummy.reshuffled:
            self.reshuffled = True

        game_done = game_done or self.reshuffled

        return self._get_obs(), reward, game_done, {}

    def _get_obs(self) -> Tuple[int, int, bool]:
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
