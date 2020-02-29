# Filename : Target_Creator.py
# Date : 2019/10/01 11.24
# Project: Dynamic Pricing
# Author : Stefano Valladares
from scipy import interpolate as interpolate
import numpy as np
import re


class Target_Creator:
    """
    This class, given a set of prices and a conversion rate for each
    price, creates the target for the simulation environment.
    This target is then used to test the algorithm and for showing in
    some plots the results obtained.
    In the case more list of conversion rates are provided, the class
    will create a demand curve and a reward curve for each set of
    conversion rates. In this way we can handle also the non stationary
    case where we have different demand curves for different phases of
    the time horizon.

    Attributes
    ----------
    - demand_curve (list of PchipInterpolator): the target demand curves
    - reward_curve (list of PchipInterpolator): the target reward curves
                                                that is made by multiply
                                                the conversion rate with
                                                its price. It represents
                                                the profit
    - prob (list): probabilities of the Bernoulli variables related to
                   the arms' rewards. Basically they are the cnv rates
                   at the price of each arm
    - arm_prices (list): list of the prices of each selcted arm
    - th (int): time horizon of the experiment
    - n_phases (int): number of phases in the time horizon

    Methods
    -------
    - demand_curve_pchip(self, prices, cnv_rates):
        Creates the target demand curves object
    - reward_curve_pchip(self, demande_curve):
        Creates the target reward curves object
    -get_demand_curve(x):
        Returns the demand curves applied to the range x
    - get_reward_curve(x):
        Returns the reward curves applied to the range x
    - get_probabilities(n_arms, demand_curve):
        Returns the set of probabilities for each arm
        and the corresponding prices.
    - get_opt_value(prob, prices):
        Returns the best value of the target reward curve
    - get_optimum_series():
        Returns a list of length th with the optimal value
        of the reward curves
    - get_optimum_series_s_ns():
        Returns a list of length th with the optimal value
        of the reward curves
    - get_optimum_series_ctx():
        Returns a list with the optimal value for the aggregate case
        and the optimal one for the disaggregate case
    - get_curves(x):
        returns both the demand curve and the reward curve
    """

    def __init__(self, prices, cnv_rates, n_arms, th, n_phases=1):
        """
        Args:
            prices (list): list of prices
            cnv_rates (list): list of the conversion rates for each
                              price and for each demand curves, if there
                              are more than one
            n_arms (int): number of arms
            th (int): time horizon
            n_phases (int): number of phases in the time horizon
        """
        if not hasattr(cnv_rates[0], '__len__'): cnv_rates = [cnv_rates]
        self.demand_curve = self.demand_curve_pchip(prices, cnv_rates)
        self.reward_curve = self.reward_curve_pchip()
        self.prob, self.arm_prices = \
            self.get_probabilities(n_arms)
        self.th = th
        self.n_phases = n_phases

    def demand_curve_pchip(self, prices, cnv_rates):
        """
        Returns: A list of interpolation demand curves given by
                 PchipInterpolator, one for each set of conversion
                 rates provided.
        """
        return [interpolate.PchipInterpolator(prices, cnv_rate)
                for cnv_rate in cnv_rates]

    def reward_curve_pchip(self):
        """
        The conversion rate is the output of the demand function given a
        price.
        The formula to get a point of the reward function given a price
        and the corresponding conversion rate is:

                reward = price * conversion_rate

        Returns: A list of reward curves, one per demand curve, given a
                 set of discrete points of the corresponding demand
                 curve.
        """
        i = np.arange(0, 10, 0.001)
        return [interpolate.PchipInterpolator(i, i * d_curve(i))
                for d_curve in self.demand_curve]

    def get_demand_curve(self, x):
        """
        Args:
            x (range): the x range where the curve must be plot

        Returns: A list of demand curves.
        """
        return [d(x) for d in self.demand_curve]

    def get_reward_curve(self, x):
        """
        Args:
            x (range): the x range where the curve must be plot

        Returns: A list of reward curves.
        """
        return [r(x) for r in self.reward_curve]

    def get_probabilities(self, n_arms):
        """
        The function, from the demand curve in input, selects n_arms
        equidistant points on the x axis. Those will be the
        probabilities of the Bernoulli variables in the Environment.
        If we have more demand curves, the function returns a list of
        set of probabilities, one for each demand curve.
        In this case they represent the probabilities of the Bernoulli
        variables in the NS_Environment.

        Args:
            n_arms (int): the number of arms

        Returns: A numpy matrix of probabilities, where each row
                 corresponds to the conversion rates of the Bernoulli
                 variables simulating the user for one demand curve,
                 and the corresponding prices.
        """
        prob = []
        prices = [x for x in np.arange(10 / n_arms, 10, 10 / (n_arms + 1))]

        for d in self.demand_curve:
            temp = []
            for price in prices:
                temp.append(d(price))
            prob.append(temp)

        return np.array(prob), prices

    def get_opt_value(self):
        """
        The function returns the optimal value of each reward curve.
        As optimal we consider the highest point in the reward curve.
        In the demand curve this can be calculated by considering the point
        that multiplied for its price has the highest value.

        Returns: A list with the optimal value of each reward curve
        """
        optimum_per_curve = []

        for p in self.prob:
            opt = np.argmax([p[i] * self.arm_prices[i]
                             for i in range(len(self.arm_prices))])
            optimum_per_curve.append(p[opt] * self.arm_prices[opt])

        return np.array(optimum_per_curve)

    def get_optimum_series(self, env_type):
        """
        Args:
            env_type (str): the environment type

        Returns: The optimum series in the contextual and
                 non contextual cases.
        """
        return self.get_optimum_series_s_ns(self.get_opt_value()) \
            if not re.match('^ctx.*$', env_type) \
          else self.get_optimum_series_ctx(self.get_opt_value())

    def get_optimum_series_s_ns(self, opt_values):
        """
        Returns: A list containing a vector of length equal to the time
                 horizon, with the optimal value of each phase repeated
                 n times, where n is the phase length.
                 In the stationary case the phase is only one
        """
        opt_per_round = np.zeros(self.th)
        phase_len = int(self.th / self.n_phases)

        for i in range(self.n_phases):
            lowest = i * phase_len
            highest = (i + 1) * phase_len
            opt_per_round[lowest:highest] = opt_values[i]

        return [opt_per_round]

    def get_optimum_series_ctx(self, opt_values):
        """
        Returns: A list of vectors of length equal to the time horizon.
                 The first with the optimal value of the aggregate
                 curve, the second with the optimal value of the
                 disaggregate curves, that is the average of the optimal
                 of each disaggregate curve
        """
        opt_per_round = [np.zeros(self.th) for i in range(2)]

        opt_values = [opt_values[0:self.n_phases],
                      [np.mean(opt_values[self.n_phases + i::self.n_phases])
                       for i in range(self.n_phases)]]

        for i in range(2):
            opt_per_round[i] = self.get_optimum_series_s_ns(opt_values[i])[0]

        return opt_per_round

    def get_curves(self, x):
        """
        Args:
            x (range): the x range where the curve must be plot

        Returns: A list containing the reward curves and the demand
                 curves of the target
        """
        return [self.get_reward_curve(x),
                self.get_demand_curve(x)]
