import unittest
from gameRL.game_simulators.blackjack import (
    BlackjackDeck,
    BlackjackCustomEnv,
)
from gameRL.game_simulators.blackjack_count import BlackjackEnvwithRunningCount, BlackjackDeckwithCount


class TestGameSimulator(unittest.TestCase):
    def test_rand_game(self):
        env = BlackjackEnvwithRunningCount(3, natural_bonus=True)
        obs = env.reset()
        env.step(2)  # player joins game
        done = False
        while not done:
            action = 0
            obs, rewards, done, info = env.step(action)
            # env.render()

    def testBlackjackDeckCountingHiLo(self):
        deck1 = BlackjackDeckwithCount(N_decks=1)
        deck3 = BlackjackDeckwithCount(N_decks=3)
        num_cards = 10
        for _ in range(num_cards):
            expected = deck1.get_running_count()
            card1, reshuffled = deck1.draw_card()
            if card1 in [2, 3, 4, 5, 6]:
                expected += 1
            elif card1 in [7, 8, 9]:
                expected += 0
            else:
                expected += -1
            actual = deck1.get_running_count()
            self.assertEqual(actual, expected, "Single Deck Counting Failed")
        for _ in range(num_cards):
            expected = deck3.get_running_count()
            card3, reshuffle = deck3.draw_card()
            if card3 in [2, 3, 4, 5, 6]:
                expected += 1
            elif card3 in [7, 8, 9]:
                expected += 0
            else:
                expected += -1
            actual = deck3.get_running_count()
            self.assertEqual(actual, expected, "Multiple Deck Counting Failed")

    def testReset(self):
        env = BlackjackEnvwithRunningCount(1)
        env.reset()
        self.assertEqual(len(env.dealer.hand), 2, "Incorrect Dealer Hand")
        self.assertEqual(len(env.dummy.hand), 2, "Incorrect Dummy Hand")

    def testRedealObserving(self):
        env = BlackjackEnvwithRunningCount(1)
        prev_dealer = env.dealer.hand.copy()
        prev_dummy = env.dummy.hand.copy()
        env.redeal()
        self.assertNotEqual(env.dealer.hand, prev_dealer, "Incorrect Dealer Hand")
        self.assertNotEqual(env.dummy.hand, prev_dummy, "Incorrect Dummy Hand")

    def testRedealJoined(self):
        env = BlackjackEnvwithRunningCount(1)
        env.step(2)
        prev_dealer = env.dealer.hand.copy()
        prev_dummy = env.dummy.hand.copy()
        prev_player = env.player.hand.copy()
        env.redeal()
        self.assertNotEqual(env.dealer.hand, prev_dealer, "Incorrect Dealer Hand")
        self.assertNotEqual(env.dummy.hand, prev_dummy, "Incorrect Dummy Hand")
        self.assertNotEqual(env.player.hand, prev_player, "Incorrect Player Hand")

    def testObserve(self):
        env = BlackjackEnvwithRunningCount(1)
        self.assertTrue(env.observing, "Default stating state is observing")
        env.step(3)
        self.assertTrue(env.observing, "Player should be observing")

    def testReshuffledTermination(self):
        env = BlackjackEnvwithRunningCount(1)
        game_done = False
        while not game_done:
            obs, _, game_done, _ = env.step(3)
        self.assertTrue(env.reshuffled)

    def testReshuffledatRhoOne(self):
        rho = 1
        env = BlackjackEnvwithRunningCount(1, rho=rho)
        game_done = False
        while not game_done:
            obs, _, game_done, _ = env.step(3)
        self.assertTrue(env.reshuffled)
        expected = 52
        self.assertEqual(env.blackjack_deck._get_cards_used(), expected)

    def testReshuffledatRhoQuarter(self):
        rho = 0.25
        env = BlackjackEnvwithRunningCount(2, rho=rho)
        game_done = False
        while not game_done:
            obs, _, game_done, _ = env.step(3)
        self.assertTrue(env.reshuffled)
        expected = 26
        self.assertEqual(env.blackjack_deck._get_cards_used(), expected)

    def testJoin(self):
        env = BlackjackEnvwithRunningCount(1)
        env.step(3)  # Player stays observing
        prev_dealer = env.dealer.hand[:]
        prev_dummy = env.dummy.hand[:]
        env.step(2)  # Player joins
        self.assertNotEqual(prev_dealer, env.dealer.hand)
        self.assertNotEqual(prev_dummy, env.dummy.hand)
        expected_hand_length = 2
        self.assertEqual(len(env.player.hand), expected_hand_length)

    def testLeave(self):
        env = BlackjackEnvwithRunningCount(1)
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
        env = BlackjackEnvwithRunningCount(1)
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
        env = BlackjackEnvwithRunningCount(1)
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

    def testObservingInvalidAction(self):
        env = BlackjackEnvwithRunningCount(1)
        env.step(2)
        env.step(3)
        prev_dealer_hand = env.dealer.hand[:]
        prev_dummy_hand = env.dummy.hand[:]
        expected_reward = 0
        _, reward, _, _ = env.step(1)  # Player hits invalidly
        self.assertNotEqual(env.dealer.hand, prev_dealer_hand)
        self.assertNotEqual(env.dummy.hand, prev_dummy_hand)
        self.assertEqual(reward, expected_reward)
        prev_dealer_hand = env.dealer.hand[:]
        prev_dummy_hand = env.dummy.hand[:]
        _, reward, _, _ = env.step(0)  # Player sticks invalidly
        self.assertEqual(reward, expected_reward)
        self.assertNotEqual(env.dealer.hand, prev_dealer_hand)
        self.assertNotEqual(env.dummy.hand, prev_dummy_hand)

    def testDoubleDown(self):
        env = BlackjackEnvwithRunningCount(1)
        env.step(4)

    def test_reset_env(self):
        env = BlackjackEnvwithRunningCount(3, natural_bonus=True)
        obs = env.reset()
        env.step(2)  # player joins game
        done = False
        for _ in range(2000):
            while not done:
                action = 0
                obs, rewards, done, info = env.step(action)
            done = False
            env.reset()
            dealer_hand = env.dealer.hand
            expected = 2
            dummy_hand = env.dummy.hand
            self.assertEqual(len(dealer_hand), expected)
            self.assertEqual(len(dummy_hand), expected)
            # env.render()

if __name__ == "__main__":
    unittest.main(verbosity=2)