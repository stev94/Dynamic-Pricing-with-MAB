# Filename : NS_TS_Learner.py
# Date : 2019/10/01 11.24
# Project: Dynamic Pricing
# Author : Stefano Valladares

from ..Stationary.TS_Learner import *
from .Sliding_Window_Handler import *


class NS_TS_Learner(TS_Learner):
    """
    This class implements the NS-TS algorithm to learn a demand curve of
    a product sold on the internet in a non stationary environment.
    In this case the demand curve can change during time, so the
    algorithm must use only the last k samples to predict each time what
    is the best arm to pull.

    The class inherits from the TS_Learner class the fundamental
    structure of TS algorithm used to solve the problem.

    Attributes
    ----------
    - Attributes of the super class
    - sw_handler (Sliding_Window_Handler): this handler controls the
                                           sliding window and gives the
                                           correct list of rewards for
                                           each arm at each time step
    Methods
    -------
    - select_arm():
        Select the arm to pull
    - update_observations(pulled_arm, reward):
        Store the observed reward
    """

    def __init__(self, n_arms, window_size, prices):
        """
        The window size variable is used to compute to include only
        recent observation in the arms' parameters estimation.

        Args:
            n_arms (int): Number of arms
            window_size (int): Size of the sliding window
            prices (list): list of prices for each arm
        """
        super().__init__(n_arms, prices)
        self.sw_handler = Sliding_Window_Handler(window_size, n_arms)

    def update_observations(self, pulled_arm, reward):
        """
        Updates the probability distribution of the arms' rewards.
        At first the observation is collected and the window is moved
        one step ahead.
        Then all the alpha and beta parameters of arms are updated,
        taking into account only the rewards obtained from the interval
        (-windows_size, t-1).

        Args:
            pulled_arm (int): The current pulled arm
            reward (int): The obtained reward after pulling the arm
        """
        super(TS_Learner, self).update_observations(pulled_arm, reward)

        arms_samples, n_samples_per_arm = \
            self.sw_handler.update_window(pulled_arm, reward, self.t)

        for arm in range(self.n_arms):
            self.beta_parameters[arm, 0] = 1 + np.sum(arms_samples[arm])
            self.beta_parameters[arm, 1] = 1 + n_samples_per_arm[arm] \
                                           - np.sum(arms_samples[arm])
