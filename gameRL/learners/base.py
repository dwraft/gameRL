import abc
import multiprocessing
import sys
from concurrent import futures
from typing import List, Set, Tuple

import torch
from torch import nn


class Agent:
    """Abstract base class for an AI agent that plays a trick taking game."""

    def __init__(self, game, player_number: int):
        self._game = game
        self._player = player_number

    @abc.abstractmethod
    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        """
        Handle an observation from the environment, and update any personal records/current
        belief/etc.

        :param action: tuple of the player who moved and the index of the card they played
        :param observation: the observation corresponding to this player as returned by the env
        :param reward: an integral reward corresponding to this player as returned by the env
        :return: None
        """
        pass

    @abc.abstractmethod
    def act(self, epsilon: float = 0):
        """
        Based on the current observation/belief/known state, select a Card to play.
        :return: the card to play
        """
        pass


class Learner:
    """Abstract base class for an AI that learns to play trick taking games."""
    def __init__(self, threading: bool = None):
        is_linux = sys.platform == "linux" or sys.platform == "linux2"
        self._use_thread = threading if threading is not None else is_linux
        self.executor = None
        if self._use_thread:
            torch.multiprocessing.set_start_method('spawn')  # allow CUDA in multiprocessing
            num_cpus = multiprocessing.cpu_count()
            num_threads = int(num_cpus / 2)  # can use more or less CPUs
            self.executor = futures.ProcessPoolExecutor(max_workers=num_threads)

    @abc.abstractmethod
    def train(self):
        """
        Given a list of trick taking game environment classes, train on them.

        :param tasks: List of environment classes, which inherit from TrickTakingGame
        :return: Trained model
        """
        pass
