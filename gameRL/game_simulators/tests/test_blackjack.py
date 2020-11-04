import unittest
from gameRL.game_simulators.blackjack import BlackjackDeck, BlackjackCustomEnv


class TestGameSimulator(unittest.TestCase):
    def testBlackjackDeckCountingHiLo(self):
        deck1 = BlackjackDeck(N_decks=1)
        deck3 = BlackjackDeck(N_decks=3)
        num_cards = 10
        for _ in range(num_cards):
            expected = deck1.get_count()
            card1 = deck1.draw_card()
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
            card3 = deck3.draw_card()
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
        self.assertFalse(env.observing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
