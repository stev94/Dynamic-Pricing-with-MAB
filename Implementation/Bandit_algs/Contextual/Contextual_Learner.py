# Filename : Contextual_TS.py
# Date : 2019/10/18 14.12
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Stationary.TS_Learner import *
from Bandit_algs.Stationary.UCB1_Learner import *
import re


class Contextual_Learner:
    """
    This class implements the a general Contextual Learner to learn a
    demand curve of a product sold on the internet in the case where
    also contexts are considered into the problem. In this case, users
    are differentiated into different contexts and the algorithm will
    try to learn a different demand curve for each context.

    This class instantiates a different learner for each context plus
    another that learnes the aggregate demand curve without considering
    the contexts.
    In this way, the algorithm can follow the aggregate learner at the
    beginning, since with a few number of observations the contexts
    learners will predict rather poorly. When the number of points is
    enough, the algorithm will split and will consider the contexts
    learner to decide which price to propose at every time step.
    The decision of split between learners is made by calculating a
    lower bound on the probability that the sum of the estimated mean
    rewards of each best arms for each context is more than the reward
    of the best arm of the aggregate learner.

    The contextual learner can follow the UCB1 policy or the Thompson
    Sampling policy. It depends from the learner_type, a string passed
    as argument in the constructor, specifying the type of the
    algorithm, as the following r.e. explains:
                                ['(ctx|a|d)_(ths|ucb)']
    The cases a and d represents two extreme cases of the contextual
    algorithm, in the 'a' case the algorithm will follow only the
    aggregate curve for all the time horizon, while in the 'd' case
    the algorithm will follow only the disaggregates curves.

    Attributes
    ----------
    - agg_learner (Learner): the learner for the aggregate curve
    - dis_learner (list): list of learners, one per context curve
    - contexts_series (list): list of contexts during one experiment
    - time_of_splitting (int): splitting time
    - splitting (bool): true if the algorithm is using the disaggregate
                        learners
    - n_contexts (int): number of contexts

    Methods
    -------
    - select_arm(context):
        Select the arm to pull, given a context
    - update_observations(pulled_arm, reward, context):
        Store the observed reward, given a context
    - check_splitting(self):
        Check the splitting condition
    - calculate_lower_bounds(self, n_samples, mean, t):
        Returns the lower bounds on the disaggregate mean rewards
    - get_estimates_mean(arm):
        Returns the estimated mean of the arm in input for all the
        learners
    - get_estimates_bounds(arm):
        Returns the upper confidence bound of the arm's reward
        for all the learners
    - get_total_rewards():
        Returns the total collected rewards
    - get_time_of_splitting():
        Returns the time when the contextual learner starts to use the
        disaggregate learners
    - select_learners(learner_type, n_ctx, prices, n_arms):
        Returns the instances of the learners basing on the learner_type
    """

    def __init__(self, n_contexts, n_arms, prices, learner_type):
        """
        Args:
            n_contexts (int): number of contexts
            n_arms (int): number of arms
            prices (list): list of prices for each arm
            learner_type (str): type of the algorithm
        """
        self.agg_learner, self.dis_learner = \
            self.select_learners(learner_type, n_contexts, prices, n_arms)
        self.contexts_series = []
        self.time_of_splitting = 0
        self.splitting = True if re.match('^d_.*$', learner_type) else False
        self.n_contexts = n_contexts

    def select_arm(self, context):
        """
        The function simply returns the best arm for both, the aggregate
        learner
        and the disaggregate learner of the context in input.

        Args:
            context (int): the contex of the current round

        Returns: The best arm so far for both, the aggregate and
                 disaggregate case.
        """
        return self.agg_learner.select_arm(), \
               self.dis_learner[context].select_arm()

    def update_observations(self, arm, reward, context):
        """
        The function calls the update method of both, the aggregate
        learner and the disaggregate learner of the context in input,
        after having store the current context.

        Args:
            arm (int): the pulled arm
            reward (int): the obtained reward after pulling the arm
            context (int): the context of the current round
        """
        self.contexts_series.append(context)

        return self.agg_learner.update_observations(arm, reward), \
               self.dis_learner[context].update_observations(arm, reward)

    def check_splitting(self):
        """
        This function calculates the condition of splitting and if it's
        verified, it stores the current time step and turn the boolean
        splitting to True.
        """
        aggr_arm = self.agg_learner.select_arm()
        disaggr_arms = [self.dis_learner[i].select_arm()
                        for i in range(self.n_contexts)]

        aggr_lb = self.calculate_lower_bounds(
            self.agg_learner.arm_counters[aggr_arm],
            self.agg_learner.get_estimates_mean(aggr_arm),
            self.agg_learner.t
        ) * self.agg_learner.prices[aggr_arm]

        disaggr_lb = np.sum(
            [self.calculate_lower_bounds(
                self.dis_learner[i].arm_counters[disaggr_arms[i]],
                self.dis_learner[i].get_estimates_mean(disaggr_arms[i]),
                self.agg_learner.t
        ) * self.agg_learner.prices[disaggr_arms[i]]
             for i in range(self.n_contexts)])

        if disaggr_lb * (1 / self.n_contexts) >= aggr_lb:
            self.splitting = True
            self.time_of_splitting = self.agg_learner.t

    def calculate_lower_bounds(self, n_samples, mean, t):
        """
        Args:
            n_samples (int): the number of samples for an arm
            mean (float): the estimated mean of an arm
            t (int): the current time step

        Returns:
        """
        return mean - ((np.sqrt(2 * np.log(t)) / n_samples)
                       if n_samples > 0 else sys.maxsize)

    def get_estimates_mean(self, arm):
        """
        Args:
            arm (int): the arm for which we want the estimated mean

        Returns: The estimated mean of the arm a for all the learners
        """
        return [self.agg_learner.get_estimates_mean(arm)] + \
               [l.get_estimates_mean(arm) for l in self.dis_learner]

    def get_estimates_bounds(self, arm):
        """
        Args:
            arm (int): the arm for which we want the confidence bound

        Returns: The estimated confidence bound of the arm a for all
                 the learners
        """
        return [self.agg_learner.get_estimates_bounds(arm)] + \
               [l.get_estimates_bounds(arm) for l in self.dis_learner]

    def get_total_rewards(self):
        """
        The function returns the total rewards obtained in one
        experiment.
        The total rewards are the same for all the learners.

        Returns: The total rewards of the contextual learner in one
                 experiment
        """
        return self.agg_learner.get_total_rewards()

    def get_time_of_splitting(self):
        """
        If the algorithm doesn't split during the experiment, it is set
        to the length of the experiment.

        Returns: The time when the contextual learner starts to use
                 the disaggregate learners.
        """
        return self.time_of_splitting \
            if self.time_of_splitting != 0 \
          else len(self.contexts_series) + 1

    def select_learners(self, learner_type, n_ctx, prices, n_arms):
        """
        Args:
            learner_type (str): the type of learner to instantiate
            n_ctx (int): the number of contexts
            prices (list): list of prices for each arm
            n_arms (int): number of arms

        Returns: The set of learners of the type in input.
        """
        return (TS_Learner(n_arms, prices), [TS_Learner(n_arms, prices)
                                             for _ in range(n_ctx)]) \
            if re.match('^(ctx|a|d)_ths$', learner_type) \
          else (UCB1_Learner(n_arms, prices), [UCB1_Learner(n_arms, prices)
                                               for _ in range(n_ctx)])
