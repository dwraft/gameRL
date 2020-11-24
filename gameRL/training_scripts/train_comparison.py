# Created by Patrick Kao
import itertools
from pathlib import Path

import stable_baselines
from stable_baselines import DQN, A2C, ACER, ACKTR, PPO2, TRPO
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

from gameRL.game_simulators.blackjack_count import BlackjackEnvwithCount

env = BlackjackEnvwithCount(3, natural_bonus=True)
TIMESTEPS_PER_MODEL = 1000000
RHO_TO_TRY = [0.25, 0.75, 0.95]
DECKS_TO_TRY = [1, 3, 10]
MAX_HAND_SUM_TO_TRY = [19, 21, 24]

# for each model, name of mode, model
models_to_train = [
    ("dqn", lambda use_env: DQN(stable_baselines.deepq.policies.MlpPolicy, use_env,
                                tensorboard_log="./runs/dqn/")),
    ("a2c", lambda use_env, log_name: A2C(MlpPolicy, use_env, tensorboard_log=log_name)),
    ("acer", lambda use_env, log_name: ACER(MlpPolicy, use_env, tensorboard_log=log_name)),
    ("acktr", lambda use_env, log_name: ACKTR(MlpPolicy, use_env, tensorboard_log=log_name)),
    ("ppo2", lambda use_env, log_name: PPO2(MlpPolicy, use_env, tensorboard_log=log_name)),
    ("trpo", lambda use_env, log_name: TRPO(MlpPolicy, use_env, tensorboard_log=log_name))
]

Path("saved_models").mkdir(parents=True, exist_ok=True)
for (name, model_gen), rho, num_decks, max_hand_sum in itertools.product(models_to_train,
                                                                         RHO_TO_TRY,
                                                                         DECKS_TO_TRY,
                                                                         MAX_HAND_SUM_TO_TRY):
    log = f"./runs/{name}/sum_{max_hand_sum}/rho_{rho}_nd_{num_decks}"
    env = BlackjackEnvwithCount(num_decks, natural_bonus=True, rho=rho, max_hand_sum=max_hand_sum)
    model = model_gen(env, log)

    model.learn(total_timesteps=TIMESTEPS_PER_MODEL)
    # test game
    reward, std = evaluate_policy(model, env, n_eval_episodes=1000)
    print(
        f"Average reward for model {name} with: rho={rho}, num decks={num_decks}, max hand sum="
        f"{max_hand_sum}: {reward}")
    # save
    model.save(f"saved_models/{name}")

env.close()
