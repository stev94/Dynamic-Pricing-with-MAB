# Filename : Bandit.py
# Date : 2019/10/01 10.51
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Stationary.Greedy_Learner import *
from Bandit_algs.Non_Stationary.NS_UCB1_Learner import *
from Bandit_algs.Non_Stationary.NS_TS_Learner import *
import re


class Bandit:
    """
    This class is used to run an arbitrary Bandit algorithm, in both
    stationary and non stationary cases. It collects the rewards and the
    confidence intervals of the arms at each experiment and it has
    methods to returns the estimates mean, the reward and the regret of
    the algorithm calculated over a set of experiments.
    In fact, given that we are simulating the environment, we need to
    make an average over an enough big number of experiments to reduce
    the noise due to the simulation.

    Arguments
    ---------
    - n_arms (int): the number of arms
    - prices (list): list of the prices of each arm
    - time_horizon (int): the time horizon of each experiment
                          (default 1000)
    - learner_type (string): the kind of policy (algorithm) to apply
    - rewards_per_experiment (list): list of the rewards obtained in
                                     each experiments
    - rewards_per_exp_per_arm (list): list of the mean of each arm at
                                      the end of each experiment
    - conf_int_per_exp_per_arm (list): list of the confidence interval
                                       of each arm at the end of each
                                       experiment
    - ws (int): the length of the window size for the stationary case
    - learners (dict): a dictionary that contains the constructor of the
                       available policies as values and the defined name
                       of the policy as key

    Methods
    -------
    - execute(env):
        Executes one experiment given the environment as input
        and save the rewards obtained
    - take_observation(learner, env):
        Gets one observation by pulling an arm and update the
        estimated values
    - save_rewards(learner):
        Saves the estimated values at the end of the experiments
        and the obtained rewards at each time step
    - get_reward_in_time():
        Returns the mean of the rewards obtained at each time step
        over the set of experiments
    - get_regret_in_time(opt_value):
        Returns the mean of the reget obtained at each time step
        over the set of experiments
    - get_arms_estimates():
        Returns the mean of the arms' estimates over the set of
        experiments
    - get_arms_cb_estimates():
        Returns the mean of the arms' confidence bounds over the set of
        experiments
    - select_learner():
        Returns the correct learner w.r.t the learner type passed in the
        constructor
    """
    def __init__(self, n_arms, prices, learner_type,
                 time_horizon=1000, ws=None):
        """
        Args:
            n_arms (int): number of arms
            prices (list): list of prices one for each arm
            learner_type (str): policy to apply
            time_horizon (int): the time horizon of each experiment
            ws (int): the window size in the nonstationary case
        """
        self.n_arms = n_arms
        self.prices = prices
        self.time_horizon = time_horizon
        self.learner_type = learner_type
        self.rewards_per_experiment = []
        self.rewards_per_exp_per_arm = [[] for i in range(n_arms)]
        self.conf_int_per_exp_per_arm = [[] for i in range(n_arms)]
        self.ws = ws
        self.learners = {'ths': TS_Learner,
                         'ucb': UCB1_Learner,
                         'greedy': Greedy_Learner,
                         'ns_ths': NS_TS_Learner,
                         'ns_ucb': NS_UCB1_Learner}

        if ws is None and learner_type in ['ns_ucb, ns_ths']:
            raise Exception("The windows size must be set for ns bandits")

    def execute(self, env):
        """
        This function executes a single experiment using a Bandit
        algorithm.
        Firstly it chooses the policy to apply and, after having seen
        all the observation, it stores the collected rewards and the
        estimates.

        Args:
            env (Environment): the simulation environment to use to
                               run the Bandit
        """
        learner = self.select_learner()

        for _ in range(self.time_horizon):
            self.take_observation(learner, env)

        self.save_rewards(learner)

    def take_observation(self, learner, env):
        """
        This function is the core of a Bandit algorithm and performs
        a round of the algorithm.

        Args:
            learner (Learner): the Bandit policy to apply
            env (Environment): the simulated environment
        """
        pulled_arm = learner.select_arm()
        reward = env.round(pulled_arm)
        learner.update_observations(pulled_arm, reward)

    def save_rewards(self, learner):
        """
        Save the collected rewards at the end of the experiment

        Args:
            learner (Learner): the Bandit policy to apply
        """
        for i in range(self.n_arms):
            self.rewards_per_exp_per_arm[i].\
                append(learner.get_estimates_mean(i))
            self.conf_int_per_exp_per_arm[i].\
                append(learner.get_estimates_bounds(i))

        self.rewards_per_experiment.append(learner.get_total_rewards())

    def get_reward_in_time(self):
        """
        Returns: The mean of the rewards obtained at each time step
                 in each experiments.
        """
        return np.mean(self.rewards_per_experiment, axis=0)

    def get_regret_in_time(self, opt_value):
        """
        In the stationary case the number of phase is just one, so the
        optimal value is the same for all the time horizon when
        calculating the regret.
        In the non stationary case, instead, it is calculated using the
        optimal value of the target reward curve for each phase and at
        each time step the rewards obtained is subtract from the optimal
        value of that phase.
        An average of the regrets for each experiments is done
        before calculating the cumulative regret returned.

        Args:
            opt_value (list): the optimal values of the reward curves,
                              one per phase

        Returns: The cumulative regret in time.
        """
        reward = self.get_reward_in_time()
        phase_len = self.time_horizon / len(opt_value)
        regret = []

        for t in range(self.time_horizon):
            phase = int(t / phase_len)
            regret.append(opt_value[phase] - reward[t])

        return np.cumsum(regret)

    def get_arms_estimates(self):
        """
        Returns: the mean over all the experiments of the estimates of
                 each arm.
        """
        return np.mean(self.rewards_per_exp_per_arm, axis=1)

    def get_arms_cb_estimates(self):
        """
            Returns: the mean over all the experiments of the confidence
                     bounds of each arm.
        """
        return np.mean(self.conf_int_per_exp_per_arm, axis=1)

    def select_learner(self):
        """
        Function from differentiating between the stationary and non
        stationary policies.

        Returns: An instance of the selected learner.
        """
        return self.learners[self.learner_type](self.n_arms, self.prices) \
            if not re.match('^ns.*$', self.learner_type) \
          else self.learners[self.learner_type](self.n_arms, self.ws,
                                                self.prices)

