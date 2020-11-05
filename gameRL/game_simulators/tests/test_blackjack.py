import unittest
from gameRL.game_simulators.blackjack import BlackjackDeck, BlackjackCustomEnv


class TestGameSimulator(unittest.TestCase):
    def test_rand_game(self):
        env = BlackjackCustomEnv(3, natural_bonus=True)
        obs = env.reset()
        env.step(2)  # player joins game
        done = False
        while not done:
            action = 0
            obs, rewards, done, info = env.step(action)
            # env.render()

    def testBlackjackDeckCountingHiLo(self):
        deck1 = BlackjackDeck(N_decks=1)
        deck3 = BlackjackDeck(N_decks=3)
        num_cards = 10
        for _ in range(num_cards):
            expected = deck1.get_count()
            card1, reshuffled = deck1.draw_card()
            if card1 in [2, 3, 4, 5, 6]:
                expected += 1
            elif card1 in [7, 8, 9]:
                expected += 0
            else:
                expected += -1
            actual = deck1.get_count()
            self.assertEqual(actual, expected, "Single Deck Counting Failed")
        for _ in range(num_cards):
            expected = deck3.get_count()
            card3, reshuffle = deck3.draw_card()
            if card3 in [2, 3, 4, 5, 6]:
                expected += 1
            elif card3 in [7, 8, 9]:
                expected += 0
            else:
                expected += -1
            actual = deck3.get_count()
            self.assertEqual(actual, expected, "Multiple Deck Counting Failed")

    def testReset(self):
        env = BlackjackCustomEnv(1)
        env.reset()
        self.assertEqual(len(env.dealer.hand), 2, "Incorrect Dealer Hand")
        self.assertEqual(len(env.dummy.hand), 2, "Incorrect Dummy Hand")

    def testObserve(self):
        env = BlackjackCustomEnv(1)
        self.assertTrue(env.observing, "Default stating state is observing")
        env.step(2)
        self.assertFalse(env.observing, "Player should not be observing")

    def testReshuffled(self):
        env = BlackjackCustomEnv(1)
        done = False
        while not done:
            obs, _, done, _ = env.step(3)
        self.assertTrue(env.reshuffled)

    def testJoin(self):
        env = BlackjackCustomEnv(1)
        env.step(3)  # Player stays observing
        prev_dealer = env.dealer.hand[:]
        prev_dummy = env.dummy.hand[:]
        env.step(2)  # Player joins
        self.assertNotEqual(prev_dealer, env.dealer.hand)
        self.assertNotEqual(prev_dummy, env.dummy.hand)
        expected_hand_length = 2
        self.assertEqual(len(env.player.hand), expected_hand_length)

    def testLeave(self):
        env = BlackjackCustomEnv(1)
        env.step(2)  # Player joins
        expected_hand_length = 2
        prev_player_hand = env.player.hand[:]
        prev_dealer_hand = env.dealer.hand[:]
        prev_dummy_hand = env.dummy.hand[:]
        self.assertEqual(len(prev_player_hand), expected_hand_length)
        self.assertEqual(len(prev_dealer_hand), expected_hand_length)
        self.assertEqual(len(prev_dummy_hand), expected_hand_length)
        env.step(3)  # Player leaves and observes
        expected_observing = True
        self.assertEqual(env.observing, expected_observing)
        self.assertIsNone(env.player)

    def testJoinandHit(self):
        env = BlackjackCustomEnv(1)
        env.step(2)  # Player joins
        expected_hand_length = 2
        prev_player_hand = env.player.hand[:]
        prev_dealer_hand = env.dealer.hand[:]
        prev_dummy_hand = env.dummy.hand[:]
        self.assertEqual(len(prev_player_hand), expected_hand_length)
        self.assertEqual(len(prev_dealer_hand), expected_hand_length)
        self.assertEqual(len(prev_dummy_hand), expected_hand_length)
        env.step(1)  # Player hits
        self.assertEqual(env.dealer.hand, prev_dealer_hand)
        self.assertEqual(env.dummy.hand, prev_dummy_hand)
        self.assertNotEqual(env.player.hand, prev_player_hand)

    def testJoinandStick(self):
        env = BlackjackCustomEnv(1)
        env.step(2)  # Player joins
        expected_hand_length = 2
        prev_player_hand = env.player.hand[:]
        prev_dealer_hand = env.dealer.hand[:]
        prev_dummy_hand = env.dummy.hand[:]
        self.assertEqual(len(prev_player_hand), expected_hand_length)
        self.assertEqual(len(prev_dealer_hand), expected_hand_length)
        self.assertEqual(len(prev_dummy_hand), expected_hand_length)
        env.step(0)  # Player sticks
        self.assertNotEqual(env.dealer.hand, prev_dealer_hand)
        self.assertNotEqual(env.dummy.hand, prev_dummy_hand)
        self.assertNotEqual(env.player.hand, prev_player_hand)


if __name__ == "__main__":
    unittest.main(verbosity=2)