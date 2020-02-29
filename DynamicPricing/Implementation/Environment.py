# Filename : Environment.py
# Date : 2019/04/21 15.29
# Project: Dynamic Pricing
# Author : Gianmarco Castro

import numpy as np


class Environment:
    """
    This class implements the simulation environment to test the
    algorithm.
    Each arm is characterized by a Bernoulli distribution with a given
    mean.
    The simulation of a new client that accepts or not a price is made
    by drawing a sample from the distribution of the arm which price
    is the one selected.

    Attributes
    ----------
    - n_arms: number of arms
    - probabilities: probabilities of the Bernoulli variables related
                     to the arms' rewards

    Methods
    -------
    - round(pulled_arm):
        Simulates the interaction of a user given a pulled arm
    """

    def __init__(self, probabilities):
        """
        Args:
            probabilities (array of floats): probabilities of the
                                             Bernoulli variables related
                                             to the arms' rewards
        """
        self.probabilities = probabilities

    def round(self, arm):
        """
        This function simulates the interaction of a user with the
        learning algorithm for a round, given a pulled arm.
        The observed reward is obtained in the simulation by drawing a
        sample from a bernoulli distribution related to an arm, with a
        given mean specified at the beginning of the algorithm.

        Args:
            arm (int): the arm to pull

        Returns: The observed reward after pulling the arm.
        """
        reward = np.random.binomial(1, self.probabilities[arm])
        return reward
