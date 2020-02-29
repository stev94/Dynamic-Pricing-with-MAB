# Filename : NS_Contextual_Env.py
# Date : 2019/10/19 23.59
# Project: Dynamic Pricing
# Author : Stefano Valladares

from ..Non_Stationary.NS_Environment import *
from .Contextual_Env import *


class NS_Contextual_Env(Contextual_Env):
    """
    This class implements the simulation environment to test the
    algorithm where also contexts about users are considered and the
    environment is non stationary.
    It inheritates from Contextual_Env the structure with the difference
    that here for each context we have an NS_Environment.

    Attributes
    ----------
    - envs: a list of NS_Environment, one per context

    Methods
    -------
    - round(arm, context):
        Simulates the interaction of a user given a pulled arm
        and a context
    """

    def __init__(self, disaggr_prob, phase_len):
        """
        Each row of the disaggr_prob matrix represents the set of
        probabilities for each context and for each arm of a single
        phase.

        Args:
            disagggr_prob (npmatrix): probabilities for each arm and
                                      for each phase
            phase_len (int): the length of a phase
        """
        self.envs = [NS_Environment(p, phase_len)
                     for p in disaggr_prob]

    def round(self, arm, context):
        """
        Select the correct NS_Environment with the context in input
        and then get the reward of that arm in that environment, so
        also considering the phase. In fact for the envs of other
        phases we also need to ensure that the time step is incremented
        by one after each observation, to make the non stationary
        simulation work.

        Args:
            arm (int): the arm to pull
            context (int): the current context

        Returns: The observed reward after pulling the arm
        """
        for e in self.envs:
            e.t += 1 if self.envs.index(e) != context else 0
        return super().round(arm, context)
