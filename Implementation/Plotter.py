# Filename : Utilities.py
# Date : 2019/05/09 03.25
# Project: Dynamic Pricing
# Author : Stefano Valladares

import matplotlib.pyplot as plt
from Target_Creator import *


class Plotter:
    """
    This class has some utilities functions to plot the results
    of the Bandits algorithm. The results are different depending on the
    Environment type, it can be one of:
                - stationary (s),
                - non stationary (ns),
                - contextual both stationary and non stationary (ctx),
    The stationary case is valid also for the AB testing.

    Attributes
    ----------
    - target (Target_Creator): variable used to create and handle the
                               target
    - curve_pos (int): column position of the first curve plot
    - axis (list): list that holds the axis for the reward and
                   demand curves
    - pos (dict): dictionary used to select the correct position size
                  based on the type of algorithm of which we are
                  plotting the results
    - figsize (dict): dictionary used to select the correct size of the
                      figure based on the type of algorithm of which we
                      are plotting the results
    - add_curves (dict): dictionary used to select the correct method
                         to plot the targets curves with the expected
                         arms' rewards of each algorithm
    - fig (Figure): the figure to plot
    - gs (GreySpec): the grid to use for the subplots

    Methods
    -------
    - plot_curves_with_points(bounds, points):
        Plots curves with arms points
    - plot_results(bandits, rewards, cb_rewards, env_type):
        Plots the results for all the Bandits passed as arguments
    - plot_in_time(self, env_type, bandits):
        Plots the regret and the reward of the algorithms in time
    - plot_curves(rewards, cb_rewards, curves, algs, title):
        Plots different curves together with the points representing the
        arms finals values
    - add_curves_ctx(rewards, cb_rewards, curves, title=''):
        Plots reward and demand curves together with the arms final
        values for the contextual case
    - split_curves(rewards, cb_rewards, curves, algs, title='')
        Plots reward and demand curves together with the arms final
        values splitting the algorithms in pairs.
    - load_curves_for_points():
        Loads the target curves
    - plot_vertical_lines(bandits):
        Plot vertical lines in the subplot.
    - plot_r_and_c_curves():
        Plots the target functions, both reward and demand curves
    """
    def __init__(self, target):
        """
        Args:
            target (Target_Creator): variable to create and handle the
                                     target
        """
        self.target = target
        self.curve_pos = 2
        self.axis = [[0, 10, -1, 3.5], [0, 10, 0, 1]]
        self.pos = {'s': (2, 3), 'ns': (2, 4), 'ctx': (2, 10)}
        self.figsize = {'s': (15, 8), 'ns': (20, 8), 'ctx': (50, 8)}
        self.add_curves = {'s':  self.plot_curves, 'ns': self.split_curves,
                           'ctx': self.add_curves_ctx}
        self.fig = None
        self.gs = None

    def plot_curves_with_points(self, points, bounds):
        """
        Args:
            bounds (list): list of the confidence intervals of each arm
                           centered on the mean, for each algorthm
            points (list): each 4-tuple in the list is composed by: the
                           vector of x points, the vector of y points,
                           the color to use for plotting the function
                           and a boolean that is true if also the
                           confidence bounds for that set of points are
                           provided in the variable bounds.
            vertical_lines (list): list of lines x = c, with c constant
        """
        plt.xlabel('Price')
        i = 0
        for point in points:
            if point[3]:
                plt.errorbar(x=point[0], y=point[1], yerr=bounds[i],
                             color="teal", capsize=3, linestyle="None",
                             marker=point[2], markersize=7, mfc="red",
                             mec="black")
                i += 1
            else:
                plt.plot(point[0], point[1], point[2], markersize=7,
                         mfc="red", mec="black")

    def plot_results(self, bandits, rewards, cb_rewards, env_type):
        """
        Plot all the Bandits results. In each algorithm there are
        always a plot for the regret in time, one for the reward in time
        and differents for the final values of the arms.
        The plots in time shows all the different Bandits together,
        while in the thers plots the Bandits are compared in pairs
        for visualization purpose.

        Args:
            bandits (dict): dictionary of Bandits
            rewards (list): list of rewards for each Bandit
            cb_rewards (list): list of confidence bounds for each
                               Bandit
            env_type (str): type of environment
        """
        if env_type == 'ab':
            env_type = 's'

        self.fig = plt.figure(figsize=self.figsize[env_type])
        self.gs = self.fig.add_gridspec(self.pos[env_type][0],
                                        self.pos[env_type][1])

        self.plot_in_time(env_type, bandits)

        curves_for_points = self.load_curves_for_points()
        self.add_curves[env_type](rewards, cb_rewards, curves_for_points,
                                  list(bandits.keys()))
        plt.show()

    def plot_in_time(self, env_type, bandits):
        """
        Args:
            env_type (str): type of environment
            bandits (dict): dictionary of Bandits
        """
        title = ['Regret', 'Reward']
        legend = [k.upper() for k in bandits.keys()] + \
                 (["Opt"] if re.match('(s|ns.*)', env_type) else
                  ["Agg opt", "Dis opt"] +
                  ["Split time: " + k.upper()
                   for k in bandits.keys() if re.match('^ctx.*$', k)])

        regret_in_time = [bandit.get_regret_in_time(self.target.get_opt_value())
                          for bandit in bandits.values()]
        reward_in_time = [bandit.get_reward_in_time()
                          for bandit in bandits.values()]
        funct_per_subplot = [regret_in_time, reward_in_time +
                             self.target.get_optimum_series(env_type)]

        for i in range(len(funct_per_subplot)):
            self.fig.add_subplot(self.gs[i, :self.curve_pos], title=title[i])
            for fun in funct_per_subplot[i]:
                plt.plot(fun)
            plt.xlabel('Time')
            plt.ylabel(title[i])

            if re.match('^ctx.*$', list(bandits.keys())[0]) and i == 1:
                self.plot_vertical_lines(bandits)
            plt.legend(legend, prop={'size': 9})

    def plot_curves(self, rewards, cb_rewards, curves, algs, title=''):
        """
        Args:
            rewards (list): list of rewards for each Bandit
            cb_rewards (list): list of confidence bounds for each Bandit
            curves (list): list of target curves
            algs (list): list of names for each algorithm to plot
            title (str): title of the sub plot
        """
        marks = ['o', 'v']
        prices = np.array(self.target.arm_prices)

        if not re.match('^.*ns.*$', algs[0]):
            legend = ["Original"] + \
                     ["Arms " + k.upper() for k in algs if k != 'greedy']
        else:
            legend = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"] + \
                     ["Arms " + k.upper() for k in algs]

        for j in range(2):
            points = list(curves[j])
            bounds = np.array(cb_rewards) * (prices if j == 0 else 1)
            for i in range(len(rewards)):
                points.append([prices + i * 0.25,
                               np.array(rewards[i])*(prices if j == 0 else 1),
                               marks[i], True])

            self.fig.add_subplot(self.gs[j, self.curve_pos],
                                 title=title + [" Rewad curve",
                                                " Demand curve"][j])
            self.plot_curves_with_points(points, bounds)
            plt.axis(self.axis[j])
            plt.ylabel(['Reward', 'Conversion rate'][j])
            plt.legend(legend, prop={'size': 9})

    def add_curves_ctx(self, rewards, cb_rewards, curves, algs):
        """
        Prepare the rewards and confidence bounds to be plotted. In
        particular this function will resort the rewards per arm for
        each context, to plot the rewards of each learner, for its
        context.

        Args:
            rewards (list): list of rewards for each Bandit
            cb_rewards (list): list of confidence bounds for each Bandit
            curves_per_context (list): list of target curves
            algs (list): list of names for each algorithm to plot
        """
        n_curves = int(len(curves) / 2)
        titles = ["Aggregate"] + ["Context " + str(n)
                                  for n in range(1, n_curves)]

        for i in range(n_curves):
            rew, cb_rew, alg = [], [], []
            x = 'd_' if i != 0 else 'a_'
            curves_per_context = [curves[i], curves[i + n_curves]]

            for j in range(len(algs)):
                if re.match('^(ctx|' + x + ').*$', algs[j]):
                    rew.append([cnv[i] for cnv in rewards[j]])
                    cb_rew.append([cb[i] for cb in cb_rewards[j]])
                    alg.append(algs[j])

            self.split_curves(rew, cb_rew, curves_per_context, alg, titles[i])

    def split_curves(self, rewards, cb_rewards, curves, algs, title=''):
        """
        This function is used to plot the rewards per algorithm for each
        arm in the case there are more than two algorithm. In fact, in
        that case they are plotted in pairs for visualization
        purposes.

        Args:
            rewards (list): list of rewards for each Bandit
            cb_rewards (list): list of confidence bounds for each Bandit
            curves (list): list of target curves
            algs (list): list of names for each algorithm to plot
            title (str): title of the sub plot
        """
        rewards, cb_rewards, algs = [rewards[:2], rewards[2:]],\
                                    [cb_rewards[:2], cb_rewards[2:]], \
                                    [algs[:2], algs[2:]]

        for i in range(len([r for r in rewards if r])):
            self.plot_curves(rewards[i], cb_rewards[i], curves, algs[i], title)
            self.curve_pos += 1

    def load_curves_for_points(self):
        """
        Load the curves for plotting the points of the arms expected
        rewards.

        Returns: A list of target functions, both reward and demand
                 curves.
        """
        curves_for_points = []
        colors = ['-b', '-m', '-g', '-y', '-k']
        x = np.arange(0, 10, 0.001)

        for c in self.target.get_curves(x):
            for i in range(0, len(c), self.target.n_phases):
                curves_for_points.append([[x, curve, color, False]
                                          for curve, color in
                                          zip(c[i:i+self.target.n_phases],
                                              colors)])
        return curves_for_points

    def plot_vertical_lines(self, bandits):
        """
        This method is used to add the splitting time in the reward
        plot for the contextual case.

        Args:
            bandits (dict): dictionary of Bandits
        """
        colors = ['b', 'r']
        vertical_lines = [bandit.get_avg_time_of_splitting()
                          for bandit in bandits.values()
                          if re.match('^ctx.*$', list(bandits.keys())[0])]
        for line, c in zip(vertical_lines, colors):
            if hasattr(line, '__len__'):
                # Non stationary
                plt.axvline(x=line[0], color=c)
                [plt.axvline(x=l, color=c, label='_nolegend_')
                 for l in line[1:]]
            else:
                # Stationary
                plt.axvline(x=line, color=c)

    def plot_r_and_c_curves(self, env_type):
        """
            Plot the target functions.
        """
        if env_type == 'ab': env_type = 's'
        curves = self.load_curves_for_points()
        fig = plt.figure(figsize=self.figsize[env_type])
        gs = fig.add_gridspec(self.pos[env_type][0], self.pos[env_type][1])
        j = 0
        legend = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        n_curves = int(len(curves) / 2)
        for i in range(len(curves)):
            fig.add_subplot(gs[int(i/n_curves), j],
                            title=[" Rewad curve",
                                   " Demand curve"][int(i/n_curves)])
            self.plot_curves_with_points(curves[i], [])
            plt.legend(legend, prop={'size': 9})
            j += 1
            if i == n_curves - 1: j = 0
        plt.show()
