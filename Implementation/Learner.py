# Filename : Learner.py
# Date : 2019/04/21 16.29
# Project: Dynamic Pricing
# Author : Gianmarco Castro

import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    """
    This class implements the abstract class of a learning algorithm.

    Attributes
    ----------
    - n_arms (int): the number of arms
    . prices (list): list of prices for each arm
    - t (int): current time step
    - arm_counters (list): list of counters counting the times each arm
                           is pulled
    - collected_rewards (list): list of all the collected rewards during
                                the run of the algorithm

    Methods
    -------
    - select_arm(): {abstract}
        Select the arm to pull at each round
    - update_observations(pulled_arm, reward):
        Store the observed reward
    - get_total_rewards():
        Return the collected rewards
    """

    def __init__(self, n_arms, prices):
        """
        Args:
            n_arms (int): Number of arms
            prices (list): list of prices for each arm
        """
        self.n_arms = n_arms
        self.prices = prices
        self.t = 0
        self.collected_rewards = np.array([])
        self.arm_counters = [0 for i in range(n_arms)]

    @abstractmethod
    def select_arm(self):
        """
        Function to select an arm base on the policy's algorithm

        Returns: The selected arm (int)
        """
        pass

    def update_observations(self, pulled_arm, reward):
        """
        This function appends the new obtained reward to the collected
        rewards list and update the arm counter of the pulled arm as
        well as the current timestep after a new observation has been
        obtained.

        Args:
            pulled_arm (int): The pulled arm
            reward (int): The observed reward after pulling the arm
        """
        self.t += 1
        self.arm_counters[pulled_arm] += 1
        self.collected_rewards = np.append(self.collected_rewards,
                                           reward * self.prices[pulled_arm])

    def get_total_rewards(self):
        """
        Returns: The list of collected rewards
        """
        return self.collected_rewards
