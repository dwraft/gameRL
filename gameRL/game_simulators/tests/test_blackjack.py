import unittest

from gameRL.game_simulators.blackjack import BlackjackCustomEnv


class TestBlackJack(unittest.TestCase):

    def test_rand_game(self):
        env = BlackjackCustomEnv(3, natural_bonus=True)
        obs = env.reset()
        for i in range(1000):
            action = 0
            obs, rewards, dones, info = env.step(action)
            env.render()

if __name__ == '__main__':
    unittest.main()