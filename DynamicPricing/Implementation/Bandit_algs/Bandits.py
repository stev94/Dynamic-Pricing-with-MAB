# Filename : Bandits.py
# Date : 2019/10/11 00.20
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Plotter import *
from Target_Creator import *
from Sequential_AB import *
from Bandit_algs.Contextual.Contextual_Bandit import *
from Bandit_algs.Contextual.NS_ContextualEnv import *
import re
import timeit


class Bandits:
    """
    This class implements a container to run a set of Bandit algorithms
    all together and plot the results in order to compare them.
    It is also in charge of creating the appropriate environment basing
    on the type of Bandits it needs to run and of building the target
    using the Target Creator and the set of conversion rates passed as
    input in the constructor.

    Attributes
    ----------
    - bandit_types (dict): dictionary used to create the different
                           Bandits that need to be compared together.
                           The index is a string that specify the type
                           of Bandit to run. It can be one of the
                           followings:
                                AB = 'ab'
                                Stationary = 's|s_ucb|s_ths'
                                Nonstationary = 'ns|ns_ucb|ns_ths'
                                Contextual = 'ctx|ctx_ucb|ctx_ths'
    - bandits_type (str): the string the specifies the type of algorithm
                          to run
    - target (Target Creator): variable to create and handle the target
                               of the simulation. It is used only for
                               plotting the results
    - env (Environment): environment of the learning algorithm. It is
                         choosen w.r.t the type of Bandit
    - bandits (dict): dictionary that holds the Bandits to run. As index
                      it uses the type of the Bandit
    - rewards (dict): dictionary to holds the expected rewards of each
                      arm for each Bandit. As index it uses the type of
                      the Bandit
    - cb_rewards (dict): dictionary to holds the confidence bounds on
                         the rewards of each arm for each Bandit.
                         As index it uses the type of the Bandit

    Methods
    -------
    - run_bandits(n_exp):
        Runs all the Bandits in the variable Bandits
    - run_bandit(bandit, n_exp):
        Runs a single Bandit and stores the rewards and the confidence
        bounds on the rewards of each arm.
    - plot_results():
        Plots the results of all the Bandits
    - select_bandits(n_arms, th, ws, n_contexts):
        Selects the correct type of Bandits to instantiate
    - select_environment(phase_len):
        Selects the correct type of Environment
    """
    def __init__(self, n_arms, prices, cnv_rates, th, bandits_type,
                 n_contexts=1, n_phases=1, ws=None):
        """
        Args:
            n_arms (int): the number of arms
            prices (list): list of prices, one for each cnv in cnv_rates
            cnv_rates (list): list of conversion rates to create the
                              target curve
            th (int): the time horizon
            bandits_type (str): the type of Bandit to instantiate
            n_contexts (int): the number fo contexts
            n_phases (int): the number of phases
            ws (int): the window size
        """

        if ws is None and n_phases == 1:
            ws = th / n_phases

        self.bandit_types = {'ab': 'ab',
                             's': ['greedy', 'ucb', 'ths'],
                             's_ucb': ['greedy', 'ucb'],
                             's_ths': ['greedy', 'ths'],
                             'ns': ['ns_ucb', 'ucb', 'ns_ths', 'ths'],
                             'ns_ucb': ['ns_ucb', 'ucb'],
                             'ns_ths': ['ns_ths', 'ths'],
                             'ctx': ['ctx_ucb', 'a_ucb', 'd_ucb',
                                     'ctx_ths', 'a_ths', 'd_ths'],
                             'ctx_ucb': ['ctx_ucb', 'a_ucb', 'd_ucb'],
                             'ctx_ths': ['ctx_ths', 'a_ths', 'd_ths'],
                             'ctx_ns': ['ctx_ns_ucb', 'a_ns_ucb', 'd_ns_ucb',
                                        'ctx_ns_ths', 'a_ns_ths', 'd_ns_ths'],
                             'ctx_ns_ucb': ['ctx_ns_ucb', 'a_ns_ucb',
                                            'd_ns_ucb'],
                             'ctx_ns_ths': ['ctx_ns_ths', 'a_ns_ths',
                                            'd_ns_ths']}
        self.bandits_type = bandits_type
        self.target = Target_Creator(prices, cnv_rates, n_arms, th, n_phases)
        self.env = self.select_env(int(th / n_phases))
        self.bandits = self.select_bandits(n_arms, th, ws, n_contexts)
        self.rewards = []
        self.cb_rewards = []

    def run_bandits(self, n_exp):
        """
        Runs all the Bandit in the dictionary bandits.

        Args:
            n_exp (int): number of experiments
        """
        print('\n' + '-'*32 + 'Starting bandits' + '-'*32)

        start = timeit.default_timer()

        for bandit in self.bandits.values():
            self.run_bandit(bandit, n_exp)

        print('\nRunning time: ' + str(timeit.default_timer() - start))

    def run_bandit(self, bandit, n_exp):
        """
        Runs a single Bandit and collects the rewards and the confidence
        bounds.
        The experiment is repeated more times to reduce the noise due
        to the simulation environment.

        Args:
            bandit (Bandit): bandit to run
            n_exp (int): number of experiments
        """
        print("Run algorithm: " + bandit.learner_type)

        for _ in range(n_exp):
            bandit.execute(self.env)

        if bandit.learner_type != 'greedy':
            self.rewards.append(bandit.get_arms_estimates())
            self.cb_rewards.append(bandit.get_arms_cb_estimates())

    def plot_results(self):
        """
        Plots the results for all the Bandits in bandits.
        """
        plotter = Plotter(self.target)
        env_type = self.bandits_type.split('_')[0]

        print('Plotting results\n')

        plotter.plot_results(self.bandits, self.rewards,
                             self.cb_rewards, env_type)

    def select_bandits(self, n_arms, th, ws, n_contexts):
        """
        The function can instantiate three types of Bandits basing on
        the bandits_type in input. The normal Bandit works for the
        stationary and non stationary case.
        The contextual is used for the contextual Bandits and finally
        the AB Learner runs an AB sequential tests.

        Args:
            n_arms (int): number of arms
            th (int): the time horizon
            ws (int): the window size
            n_contexts (int): the number of contexts

        Returns: A dictionary with all the Bandits instantiated.
        """
        if not re.match('^(ctx.*|ab)$', self.bandits_type):
            return {b_type: Bandit(n_arms, self.target.arm_prices,
                                   b_type, th, ws)
                    for b_type in self.bandit_types[self.bandits_type]}

        elif not re.match('^ab$', self.bandits_type):
            return {b_type: Contextual_Bandit(n_arms, self.target.arm_prices,
                                              b_type, n_contexts, th, ws)
                    for b_type in self.bandit_types[self.bandits_type]}
        else:
            return {self.bandit_types[self.bandits_type]:
                        Sequential_AB(th, n_arms, self.target.arm_prices)}

    def select_env(self, phase_len):
        """
        Selects the correct environment based on the bandits_type
        variable in input.

        Args:
            phase_len (int): length of a single phase for the non
                             stationary case

        Returns: One instance of the correct Environment
        """
        return Environment(self.target.prob[0]) \
            if re.match('^(s.*|ab)$', self.bandits_type) \
          else NS_Environment(self.target.prob, phase_len) \
            if re.match('^ns.*$', self.bandits_type) \
          else Contextual_Env(self.target.prob[1:]) \
            if re.match('^ctx(|.{4})$', self.bandits_type) \
          else NS_Contextual_Env([self.target.prob[4+i*4:4+(i+1)*4]
                                  for i in range(4)], phase_len)
