# Filename : Greedy_Learner.py
# Date : 2019/05/08 10.31
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Learner import *


class Greedy_Learner(Learner):

    """
    This class implements a Greedy algorithm to learn the demand curve
    of a product sold on the internet.
    The greedy policy acts by choosing always the arm with the highest
    expected mean.
    The class inherits from the Learner class the fundamental structure
    of all the Bandit_algs algorithms used to solve the problem.

    Attributes
    ----------
    - Attributes of the super class
    - expt_reward(nparray) : array that holds the current expected
                             reward of each arm.

    Methods
    -------
    - select_arm():
        Select the arm to pull
    - update_observations(pulled_arm, reward):
        Store the observed reward
    """
    def __init__(self, n_arms, prices):
        """
        Args:
            n_arms (int): Number of arms
            prices (list): list of prices for each arm
        """
        self.expt_rewards = np.zeros(n_arms)
        super().__init__(n_arms, prices)

    def select_arm(self):
        """
        The function simply selects always the arm whose reward has the
        highest current expectation.

        Returns: The best arm so far by following the UCB1 policy
        """
        if self.t < self.n_arms:
            return self.t

        idxs = np.argwhere(
            np.multiply(self.expt_rewards, self.prices) ==
            np.multiply(self.expt_rewards, self.prices).max()
        ).reshape(-1)

        return np.random.choice(idxs)

    def update_observations(self, pulled_arm, reward):
        """
        Update the expectation of the pulled arm.

        Args:
            pulled_arm (int): The pulled arm to update
            reward (int): The observed reward obtained by pulling
                          the arm
        """
        super().update_observations(pulled_arm, reward)
        self.expt_rewards[pulled_arm] = (self.expt_rewards[pulled_arm]
                                         * (self.t - 1) + reward) / self.t

    def get_estimates_mean(self, arm):
        """
        Args:
            arm (int): the arm for which we want the expected mean

        Returns: The expected mean of the arm in input
        """
        return self.expt_rewards[arm]

    def get_estimates_bounds(self, arm):
        return 0
