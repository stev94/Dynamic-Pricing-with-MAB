# Filename : Contextual_Bandit.py
# Date : 2019/10/18 15.16
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Bandit import *
from Bandit_algs.Contextual.NS_Contextual_Learner import *
import random


class Contextual_Bandit(Bandit):
    """
    This class is used to run an arbitrary Contextual Bandit algorithm,
    in both stationary and non stationary cases.
    It inherits from the Bandit class all the methods for handling a
    general Bandit.

    Arguments
    ---------
    - n_contexts (int): the number of contexts
    - contexts_per_exp (list): list that holds the series of contexts
                               per each experiment
    - time_of_splitting (list): list of the times in which the algorithm
                                splits in each phase

    Methods
    -------
    - take_observation(learner, env):
        Gets one observation by pulling an arm and update the
        estimated values
    - save_rewards(learner):
        Saves the estimated values at the end of the experiments
        and the obtained rewards at each time step
    - get_avg_time_of_splitting():
        Returns the mean time of splitting over all the experiments
    - get_regret_in_time(opt_value):
        Returns the mean of the reget obtained at each time step
        over the set of experiments
    - select_learner():
        Returns the correct learner w.r.t the learner type passed in the
        constructor
    """
    def __init__(self, n_arms, prices, learner_type, n_contexts,
                 time_horizon=1000, ws=None):
        """
        Args:
            n_arms (int): number of arms
            prices (list): list of prices one for each arm
            learner_type (str): policy to apply
            n_contexts (int): the number of contexts
            time_horizon (int): the time horizon of each experiment
            ws (int): the window size in the nonstationary case
        """
        self.n_contexts = n_contexts
        self.contexts_per_exp = []
        self.time_of_splitting = []
        super().__init__(n_arms, prices, learner_type,
                         time_horizon=time_horizon, ws=ws)

    def take_observation(self, learner, env):
        """
        This function override the one in the Bandit class. It selects
        as the parents method, but in this case we select a random
        context that the learner and the environment need to select
        the arm and the reward.
        The algorithm will select the agg_arm until the splitting
        variable in the learner is False. This is because we want the
        learner to follow the aggregate curve when the number of
        observation is low and then, splits to the disaggregates curves
        using a lower bound for taking the decision of when to split.

        Args:
            learner (Learner): the Bandit policy to apply
            env (Environment): the simulated environment
        """
        curr_context = random.randint(0, self.n_contexts - 1)

        aggr_arm, disaggr_arm = learner.select_arm(curr_context)
        arm = aggr_arm if not learner.splitting else disaggr_arm
        reward = env.round(arm, curr_context)
        learner.update_observations(arm, reward, curr_context)

        if not re.match('^[ad].*$', self.learner_type) and \
           not learner.splitting and \
               learner.agg_learner.t > self.n_arms:
            learner.check_splitting()

    def save_rewards(self, learner):
        """
        Saves the contexts occurred during the experiment and the
        splitting time. Then calls the super method.

        Args:
            learner (Learner): the Bandit policy to apply
        """
        self.time_of_splitting.append(learner.get_time_of_splitting())
        self.contexts_per_exp.append(learner.contexts_series)

        super().save_rewards(learner)

    def get_avg_time_of_splitting(self):
        """
        Returns: The average time of splitting over all the experiments.
        """
        return np.mean(self.time_of_splitting, axis=0)

    def get_regret_in_time(self, opt_value):
        """
        In this case for calculating the regret we need to take at
        each experiment, in each time step, the best value of the
        context occurred at that time and subcract it to the reward
        obtained at that time step in that experiment.

        Args:
            opt_value (list): the optimal values of the reward curves,
                              one per phase

        Returns: The cumulative regret in time.
        """
        disaggr_regret = [[0 for _ in range(self.time_horizon)]
                          for _ in range(len(self.contexts_per_exp))]

        n_phases = int(len(opt_value) / (self.n_contexts + 1))
        phase_len = self.time_horizon / n_phases

        for e in range(len(self.rewards_per_experiment)):
            for t in range(self.time_horizon):
                c = self.contexts_per_exp[e][t]
                phase = int(t / phase_len)
                disaggr_regret[e][t] = opt_value[n_phases*(c+1)+phase] \
                                       - self.rewards_per_experiment[e][t]

        return np.cumsum(np.mean(disaggr_regret, axis=0))

    def select_learner(self):
        """
         Function from differentiating between the stationary and non
         stationary policies in the contextual case.

         Returns: An instance of the selected learner.
         """
        if not re.match('^.*ns.*$', self.learner_type):
            return Contextual_Learner(self.n_contexts, self.n_arms,
                                      self.prices, self.learner_type)
        else:
            return NS_Contextual_Learner(self.n_contexts, self.n_arms,
                                         self.prices, self.learner_type,
                                         self.ws)
