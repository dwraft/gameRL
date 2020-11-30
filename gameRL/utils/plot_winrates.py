# Created by Patrick Kao
import copy
from os import listdir
from os.path import join, isfile

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import ACER, A2C, PPO2, DQN, ACKTR
from stable_baselines.common.evaluation import evaluate_policy

from gameRL.game_simulators.blackjack_count import BlackjackEnvwithCount

name_map = {"a2c": A2C,
            "acer": ACER,
            "acktr": ACKTR,
            "dqn": DQN,
            "ppo2": PPO2
            }

NUM_TO_RUN = 5000


def plot_winrates(directory, show_std=False):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    #files = filter(lambda x: x[-4] == ".zip", files)

    params = copy.deepcopy(files)
    params = [file.replace("sum_", "") for file in params]
    params = [file.replace("rho_", "") for file in params]
    params = [file.replace("nd_", "") for file in params]
    params = [file.replace(".zip", "") for file in params]
    params = np.array([file.split("_") for file in params])

    param_file_map = {tuple(param): filename for param, filename in zip(params, files)}

    combos = np.unique(params[:, 1:], axis=0)
    fig, axs = plt.subplots(len(combos))
    for i, combo in enumerate(combos):
        match_hits = params[:, 1:] == combo
        match_hits = np.all(match_hits, axis=1)
        matches = params[match_hits, :]
        names = []
        winrates = []
        stds = []
        for match in matches:
            full_filename = f"{directory}/{param_file_map[tuple(match)]}"
            model = name_map[match[0]].load(full_filename)
            env = BlackjackEnvwithCount(int(match[3]), natural_bonus=True, rho=float(match[2]),
                                        max_hand_sum=int(match[1]))
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=NUM_TO_RUN)
            names.append(match[0])
            winrates.append(mean_reward)
            stds.append(std_reward)

        names_pos = [i for i, _ in enumerate(names)]

        axs[i].bar(names_pos, winrates, yerr=stds if show_std else None)
        axs[i].set_xticks(names_pos)
        axs[i].set_xticklabels(names)
        axs[i].set_ylabel("Mean reward")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_winrates("saved_models")
