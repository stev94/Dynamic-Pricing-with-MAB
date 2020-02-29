# Filename : Sliding_Window_Handler.py
# Date : 2019/10/10 00.39
# Project: Dynamic Pricing
# Author : Stefano Valladares


class Sliding_Window_Handler:
    """
    This class implements two functions to deal with the non stationary
    environment.
    It maintains a list of rewards for each arm from which can calculate
    the part of the rewards that fall into the current sliding window.

    Arguments
    ---------
    - ws (int): the window size
    - rewards_per_arm (list): a list of rewards for each arm

    Methods
    -------
    - get_samples_of_each_arm(t):
        Returns a list with, for each arm, the list of samples
        in the current time window
    - update_window(arm, reward, t):
        Stores the collected rewards and updates the window
    """
    def __init__(self, window_size, n_arms):
        self.ws = window_size
        self.rewards_per_arm = [[] for x in range(n_arms)]

    def get_samples_of_each_arm(self, t):
        """
        This function finds the positive rewards of each arm in the last
        windows size time steps and returns them.

       Args:
            t (int): the last index of the sliding window

        Returns: A list containing the samples in the current
                 window for each arm.
        """
        return [list(i) for i in [filter(lambda x: x >= 0, row[-self.ws:t])
                                  for row in self.rewards_per_arm]]

    def update_window(self, arm, reward, t):
        """
        This function updates the sliding window and returns the new
        set of rewards one for each arm and their corresponding counter
        in the new sliding window.
        A -1 is append for all the arms that in that round have not been
        pulled, in this way we can calculate the rewards in the sliding
        windows by taking the entries in the interval
        [-sliding_window, t] and eliminating the ones that are less than
        zero. Look forward more efficient tecniques.

        Args:
            arm (int): the pulled arm
            reward (int): the obtained reward after pulling the arm
            t (int): the current time step

        Returns: The rewards in the current sw for each arm and the
                 number of times each arm has been pulled in the sw.

        """
        for r_arm in self.rewards_per_arm:
            r_arm.append(-1)
        self.rewards_per_arm[arm][-1] = reward

        arms_samples = self.get_samples_of_each_arm(t)
        n_samples_per_arm = [len(x) for x in arms_samples]

        return arms_samples, n_samples_per_arm
