# Created by Patrick Kao
import argparse
import copy
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import ACER, A2C, PPO2, DQN, ACKTR
from stable_baselines.common.evaluation import evaluate_policy

from gameRL.game_simulators.blackjack import BlackjackEnvwithCount

name_map = {"a2c" : A2C,
            "acer" : ACER,
            "acktr" : ACKTR,
            "dqn" : DQN,
            "ppo2" : PPO2
            }

NUM_TO_RUN = 5000

def plot_winrates(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    files = filter(lambda x: x[-4]==".zip", files)

    params = copy.deepcopy(files)
    params = [file.replace("sum_", "") for file in params]
    params = [file.replace("rho_", "") for file in params]
    params = [file.replace("nd_", "") for file in params]
    params = [file.replace(".zip", "") for file in params]
    params = np.array([file.split("_") for file in params])

    param_file_map = {tuple(param):filename for param,filename in zip(files, params)}

    combos = np.unique(params[:, 1:], axis=0)
    fig, axs = plt.subplots(len(combos))
    for i, combo in enumerate(combos):
        matches = params[params[:,1:]==combo]
        names = []
        winrates = []
        stds = []
        for match in matches:
            model = name_map[match[0]].load(param_file_map)
            env = BlackjackEnvwithCount(match[3], natural_bonus=True, rho=match[2],
                                        max_hand_sum=match[1])
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            names.append(match[0])
            winrates.append(mean_reward)
            stds.append(std_reward)

        names_pos = [i for i, _ in enumerate(names)]

        axs[i].bar(names_pos, winrates, yerr=stds)
        axs[i].xticks(names_pos, names)
        axs[i].ylabel("Mean reward")

    plt.show()


if __name__ == "__main__":
    plot_winrates()