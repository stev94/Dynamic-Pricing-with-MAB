# Filename : NS_Contextual_Learner.py
# Date : 2019/10/22 00.59
# Project: Dynamic Pricing
# Author : Stefano Valladares

from .Contextual_Learner import *
from ..Non_Stationary.NS_TS_Learner import *
from ..Non_Stationary.NS_UCB1_Learner import *


class NS_Contextual_Learner(Contextual_Learner):
    """
    This class implements the a general Contextual Learner to learn a
    demand curve of a product sold on the internet in the case where
    also contexts are considered into the problem and the environment is
    non stationary.
    In this case, users are differentiated into different contexts and
    the algorithm will try to learn a different set of demand curves for
    each context.
    In particular for each context there will be n different demand
    curves, one for each different phase of the environment. Then the
    alogirthm will have to adapt to changes of the environment as well
    as understand how to differentiate between users.

    This class instantiates a different nonstationary learner for each
    context plus another that learnes the aggregate demand curve without
    considering the contexts, but considering the phases.
    The decision of split between learners is made in the same way as
    the stationary case, but it is take one time per phase. Since the
    learner doesn't know how long a phase actually lasts, it will use
    the length of the sliding window as length of the phase.

    The contextual learner can follow the NS UCB1 policy or the NS
    Thompson Sampling policy. It depends from the learner_type, a string
    passed as argument in the constructor, specifying the type of the
    algorithm, as the following r.e. explains:
                        ['(ctx|a|d)_ns_(ths|ucb)']
    The cases a and d represents two extreme cases of the contextual
    algorithm, in the 'a' case the algorithm will follow only the
    aggregate curve in each phase, while in the 'd' case the algorithm
    will follow only the disaggregates curves.

    The class inherates from the Contextual Learner the basic structure
    of a Contextual Bandit policy.

    Attributes
    ----------
    - Attributes of the super class
    - ws (int): the window size
    - time_of_splittings (list): list of time of splittings, one per
                                 phase

    Methods
    -------
    - update_observations(pulled_arm, reward, context):
        Store the observed reward, given a context
    - reset_splitting():
        Reset the splitting variable for the next phase
    - get_time_of_splitting():
        Returns the times when the contextual learner starts to use the
        disaggregate learners, in each phase
    - select_learners(learner_type, n_ctx, prices, n_arms):
        Returns the instances of the learners basing on the learner_type
    """
    def __init__(self, n_contexts, n_arms, prices, learner_type, ws):
        """
        Args:
            n_contexts (int): number of contexts
            n_arms (int): number of arms
            prices (list): list of prices for each arm
            learner_type (str): type of the algorithm
            ws (int): the window size
        """
        self.ws = ws
        self.time_of_splittings = []
        super().__init__(n_contexts, n_arms, prices, learner_type)

    def update_observations(self, arm, reward, context):
        """
        The function checks if the end of the phase has been reached,
        to reset the splitting variable. Then it sum one step in all the
        disaggregate learners that are learning a context different from
        the one in input. Finally it calls the super method to finalize
        the update.

        Args:
            arm (int): the pulled arm
            reward (int): the obtained reward after pulling the arm
            context (int): the context of the current round
        """
        if (self.agg_learner.t + 1) % self.ws == 0:
            self.reset_splitting()

        for i in range(self.n_contexts):
            self.dis_learner[i].t += 1
        self.dis_learner[context].t -= 1

        return super().update_observations(arm, reward, context)

    def reset_splitting(self):
        """
        The functions stores the time of splitting of the ending
        phase and restores the splitting variable for the next phase.
        Note that in case the algorithm doesn't split during one phase
        time_of_splitting will be 0 and the function
        get_time_of_splitting will return the current time horizon,
        that will correspond to the end of the current phase.
        """
        self.time_of_splittings.append(super().get_time_of_splitting())
        self.time_of_splitting = 0
        self.splitting = False

    def get_time_of_splitting(self):
        """
        Returns: The times when the contextual learner starts to use
                 the disaggregate learners, one for each phase.
        """
        return self.time_of_splittings

    def select_learners(self, learner_type, n_ctx, prices, n_arms):
        """
        Args:
            learner_type (str): the type of learner to instantiate
            n_ctx (int): the number of contexts
            prices (list): list of prices for each arm
            n_arms (int): number of arms

        Returns: The set of non stationary learners of the selected
                 type.
        """
        if re.match('^(ctx|a|d)_ns_ths$', learner_type):
            return (NS_TS_Learner(n_arms, self.ws, prices),
                    [NS_TS_Learner(n_arms, self.ws, prices)
                     for _ in range(n_ctx)])
        else:
            return (NS_UCB1_Learner(n_arms, self.ws, prices),
                    [NS_UCB1_Learner(n_arms, self.ws, prices)
                     for _ in range(n_ctx)])
