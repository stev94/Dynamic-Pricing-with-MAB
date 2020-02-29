# Filename : NS_UCB1_Learner.py
# Date : 2019/05/08 12.58
# Project: Dynamic Pricing
# Author : Stefano Valladares

from ..Stationary.UCB1_Learner import *
from .Sliding_Window_Handler import *


class NS_UCB1_Learner(UCB1_Learner):
    """
    This class implements the NS-UCB1 algorithm to learn a demand curve
    of a product sold on the internet in a non stationary environment.
    In this case the demand curve can change during time, so the
    algorithm must use only the last k samples to predict each time what
    is the best arm to pull.

    The class inherits from the UCB_Learner class the fundamental
    structure of UCB1 algorithm used to solve.

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
    - upper_bound_update(n_samples):
        Calculate the upper confidence bound of the arm's reward
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
        Then all the upper confidence bounds and the means of arms are
        updated, taking into account only the rewards obtained from the
        interval (-windows_size, t-1).

        Args:
            pulled_arm (int): The current pulled arm
            reward (int): The obtained reward after pulling the arm
        """
        super(UCB1_Learner, self).update_observations(pulled_arm, reward)

        arms_samples, arm_counters = \
            self.sw_handler.update_window(pulled_arm, reward, self.t)

        for arm in range(self.n_arms):
            self.estimates[arm, 0] = np.mean(arms_samples[arm]) \
                if arm_counters[arm] != 0 else 0
            self.estimates[arm, 1] = self.upper_bound_update(
                arm_counters[arm] - (1 if arm == pulled_arm else 0)
            )

    def upper_bound_update(self, n_samples):
        """
        Args:
            n_samples (int): Number of samples observed

        Returns: The upper confidence bound of the reward of the arm
        """
        return 1 * np.sqrt(np.log(min(self.t, self.sw_handler.ws))
                           / n_samples) \
            if n_samples > 0 \
          else sys.maxsize
