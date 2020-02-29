# Filename : TS_Learner.py
# Date : 2019/10/01 11.24
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Learner import *
from scipy.stats import beta


class TS_Learner(Learner):
    """
    This class implements the Thompson Sampling algorithm to learn a
    demand curve of a product sold on the internet.
    The TS policy maintaing a couple of variable for each arm
    (alpha, beta). Then at each iteration it samples from the beat
    distribution of each arm and selects the arm with the highest
    sample. In our case the sample is then multiply by the arm's price
    because we want to learn the reward curve.
    The class inherits from the Learner class the fundamental structure
    of all the Bandit_algs algorithms used to solve the problem.

    Attributes
    ----------
    - Attributes of the super class
    - beta_parameters (npmatrix) : matrix (n_arms x 2) in which for each
                                   arm we have the alpha and beta
                                   parameters of each arm.

    Methods
    -------
    - select_arm():
        Selects the arm to pull
    - update_observations(pulled_arm, reward):
        Stores the observed reward
    - get_estimates_mean(arm):
        Returns the estimated mean of the arm in input
        calculated from its beta distribution
    - get_estimates_bounds(arm):
        Returns the upper confidence bound of the arm's reward,
        calculated from its beta distribution
    """

    def __init__(self, n_arms, prices):
        """
        Args:
            n_arms (int): Number of arms
            prices (list): list of prices for each arm
        """
        self.beta_parameters = np.ones((n_arms, 2), float)
        super().__init__(n_arms, prices)

    def select_arm(self):
        """
        The arm is chosen by firstly sample from the beta distribution
        of each arm with the updated parameters. The each sample is
        multiplied by the arm's price and finally the arm with the
        highest value is returned.

        Returns: the best arm so far by following the TS policy
        """
        samples = np.multiply(np.random.beta(self.beta_parameters[:, 0],
                                             self.beta_parameters[:, 1]),
                              self.prices)
        return np.argmax(samples)

    def update_observations(self, pulled_arm, reward):
        """
        The beta parameters of the pulled arm are updated accordingly
        the formula:
                (a, b) = (a, b) + (reward, 1 - reward)
        Args:
            pulled_arm (int): the current pulled arm
            reward (int): the obtained reward after pulling the arm
        """
        super().update_observations(pulled_arm, reward)

        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1.0 - reward

    def get_estimates_mean(self, arm):
        """
        The mean of the beta distribution is: alpha / (alpha + beta).

        Args:
            arm (int): the arm for which we want the estimated mean

        Returns: The estimated mean of the arm a
        """
        return self.beta_parameters[arm, 0] / (self.beta_parameters[arm, 0]
                                               + self.beta_parameters[arm, 1])

    def get_estimates_bounds(self, arm):
        return beta.interval(0.95, self.beta_parameters[arm, 0],
                             self.beta_parameters[arm, 1])[1]
