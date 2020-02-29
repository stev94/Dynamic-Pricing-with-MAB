# Filename : UCB1_Learner.py
# Date : 2019/05/08 09.10
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Learner import *
import sys
import random


class UCB1_Learner(Learner):
    """
    This class implements the UCB1 algorithm to learn a demand curve of
    a product sold on the internet.
    The UCB1 policy selects the arm with the highest upper bound on the
    mean estimates. As upper bound here we use the Hoeffding bound.
    The class inherits from the Learner class the fundamental structure
    of all the Bandit_algs algorithms used to solve the problem.

    Attributes
    ----------
    - Attributes of the super class
    - estimates (npmatrix) : matrix (n_arms x 2) in which for each arm
                             we have the current expected mean and the
                             respectively upper confidence bound.

    Methods
    -------
    - select_arm():
        Select the arm to pull
    - update_observations(pulled_arm, reward):
        Store the observed reward
    - upper_bound_update(n_samples):
        Calculate the upper confidence bound of the arm's reward
    - get_estimates_mean(arm):
        Returns the estimated mean of the arm in input
    - get_estimates_bounds(arm):
        Returns the upper confidence bound of the arm's reward
    """

    def __init__(self, n_arms, prices):
        """
        Args:
            n_arms (int): number of arms
            prices (list): list of prices for each arm
        """
        super().__init__(n_arms, prices)
        self.estimates = np.zeros((n_arms, 2))

    def select_arm(self):
        """
        This function looks for the best arm and return it.
        In the case of the UCB1 algorithm the best arm is the
        one that maximizes the sum between:
            - The empirical mean of the reward of an arm a at time t
              multiplied by the price of a
            - The upper confidence bound of a at t

        Returns: The best arm so far by following the UCB1 policy
        """
        upper_bounds = self.estimates.sum(axis=1, dtype=float) * self.prices
        max_values = np.argwhere(upper_bounds == np.amax(upper_bounds))
        return self.t if self.t < self.n_arms else random.choice(max_values)[0]

    def update_observations(self, pulled_arm, reward):
        """
        Update the probability distribution of the arms' rewards.
        All the upper confidence bounds and only the mean of the
        pulled arm are updated after having stored the observed reward.

        Args:
            pulled_arm (int): the  pulled arm
            reward (int): the obtained reward after pulling the arm
        """
        super().update_observations(pulled_arm, reward)

        for arm in range(self.n_arms):
            self.estimates[arm, 1] = self.upper_bound_update(
                self.arm_counters[arm] - (1 if arm == pulled_arm else 0)
            )

        l = self.arm_counters[pulled_arm]
        self.estimates[pulled_arm, 0] = (self.estimates[pulled_arm, 0]
                                         * (l - 1) + reward) / l

    def upper_bound_update(self, n_samples):
        """
        Args:
            n_samples (int): Number of samples observed

        Returns: The upper confidence bound of the reward of the arm
        """
        return 1 * np.sqrt(np.log(self.t) / n_samples) \
            if n_samples > 0 \
          else sys.maxsize

    def get_estimates_mean(self, arm):
        """
        Args:
            arm (int): the arm for which we want the estimated mean

        Returns: The estimated mean of the arm a
        """
        return self.estimates[arm, 0]

    def get_estimates_bounds(self, arm):
        """
        Args:
            arm (int): the arm for which we want the confidence bound

        Returns: The estimated confidence bound of the arm a
        """
        return self.estimates[arm, 1]
