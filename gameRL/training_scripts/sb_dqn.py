# Created by Patrick Ka
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy

from gameRL.game_simulators.blackjack import BlackjackCustomEnv

env = BlackjackCustomEnv(3, natural_bonus=True)

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./runs/dqn/")
model.learn(total_timesteps=10000)

# test game
reward, std = evaluate_policy(model, env, n_eval_episodes=100)
print(f"average reward: {reward}")

env.close()
