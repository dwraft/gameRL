# Created by Patrick Ka
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy

from gameRL.game_simulators.blackjack import BlackjackCustomEnv


def play_game(model):
    done = False
    obs = env.reset()
    rewards = None
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    return rewards


env = BlackjackCustomEnv(3, natural_bonus=True)

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

# test game
games_to_play = 1000
total_reward = 0
for i in range(games_to_play):
    total_reward += play_game(model)
print(f"total reward: {total_reward}\naverage reward: {total_reward / games_to_play}")

env.close()
