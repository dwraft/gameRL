# Created by Patrick Kao
from pathlib import Path

import stable_baselines
from stable_baselines import DQN, A2C, ACER, ACKTR, PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

from gameRL.game_simulators.blackjack import BlackjackCustomEnv

env = BlackjackCustomEnv(3, natural_bonus=True)
TIMESTEPS_PER_MODEL = 1000000
# for each model, name of mode, model
models_to_train = [
    ("dqn", DQN(stable_baselines.deepq.policies.MlpPolicy, env, tensorboard_log="./runs/dqn/")),
    ("a2c", A2C(MlpPolicy, env, tensorboard_log="./runs/a2c/")),
    ("acer", ACER(MlpPolicy, env, tensorboard_log="./runs/acer/")),
    ("acktr", ACKTR(MlpPolicy, env, tensorboard_log="./runs/acktr/")),
    ("ppo2", PPO2(MlpPolicy, env, tensorboard_log="./runs/ppo2/"))]

Path("saved_models").mkdir(parents=True, exist_ok=True)
for name, model in models_to_train:
    model.learn(total_timesteps=TIMESTEPS_PER_MODEL)
    # test game
    reward, std = evaluate_policy(model, env, n_eval_episodes=1000)
    print(f"Average reward for model {name}: {reward}")
    # save
    model.save(f"saved_models/{name}")

env.close()
