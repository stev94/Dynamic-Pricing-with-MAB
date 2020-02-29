# Filename : Content_Env.py
# Date : 2019/10/16 20.02
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Environment import *


class Contextual_Env:
    """
    This class implements the simulation environment to test the
    algorithm where also contexts about users are considered.
    Here we use a set of Environments, one for each context with
    the probabilities of the Bernoulli variable taking from the
    demand curve of that context.

    Attributes
    ----------
    - Attributes of the super class
    - envs: a list of Environment, one per context

    Methods
    -------
    - round(arm, context):
        Simulates the interaction of a user given a pulled arm
        and a context
    """

    def __init__(self, disaggr_prob):
        """
        Each row of the probability matrix represents the probabilities
        for each arm of a single context.

        Args:
            disaggr_prob (npmatrix): probabilities for each arm and
                                     for each context
        """
        self.envs = [Environment(p) for p in disaggr_prob]

    def round(self, arm, context):
        """
        Select the correct Environment with the context in input
        and then get the reward of that arm in that environment.

        Args:
            arm (int): the arm to pull
            context (int): the current context

        Returns: The observed reward after pulling the arm
        """
        return self.envs[context].round(arm)
