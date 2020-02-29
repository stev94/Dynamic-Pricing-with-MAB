# Filename : NS_Environment.py
# Date : 2019/05/08 12.21
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Environment import *


class NS_Environment(Environment):
    """
    This class implements the simulation environment to test the
    algorithm when the environment is not stationary.
    Each arm is characterized by a Bernoulli distribution with a given
    mean.
    The simulation of a new client that accepts or not a price is made
    by drawing a sample from the distribution of the arm which price
    is the one selected.
    In this case, we change the set of probability after some
    observation, this simulates the fact the now the distribution is
    changing over time.

    Attributes
    ----------
    - Attributes of the super class
    - t: current time step
    - phase_len: number of steps of a single phase (assumed all equal)

    Methods
    -------
    - round(pulled_arm):
        Simulates the interaction of a user given a pulled arm in a non
        stationary environment
    """

    def __init__(self, probabilities, phase_len):
        """
        Each row of the probability matrix represents the probabilities
        for each arm of a single phase.
        The variable t keeps track of the current round of the
        experiment. In this way we can compute at each round the
        corresponding phase of the environment.

        Args:
            probabilities (npmatrix): probabilities for each arm and
                                      for each phase
            phase_len (int): the length of a phase
        """
        super().__init__(probabilities)
        self.t = 0
        self.phase_len = phase_len

    def round(self, arm):
        """
        After selecting the current phase and the corresponding
        probability for the pulled arm, the function simulates the
        interaction with the learning algorithm by drawing a sample
        from the probability distribution chosen.
        The module operation allows us to use the same environment
        for more experiments, because in this way we don't care about t,
        the phases are thinking to repeat after having finished.

        Args:
            arm (int): the arm to pull

        Returns: The observed reward after pulling the arm
        """
        current_phase = int(self.t / self.phase_len) % len(self.probabilities)
        p = self.probabilities[current_phase][arm]
        self.t += 1
        return np.random.binomial(1, p)


