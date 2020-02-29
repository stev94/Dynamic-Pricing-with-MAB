# Filename : Input_Data.py
# Date : 2019/10/24 02.09
# Project: Dynamic Pricing
# Author : Stefano Valladares

import functools
import operator
import random
import re

import numpy as np


class Data_Provider:
    """
    This class is used to retrieve the data needed to create the
    target functions.
    There is an array of 4x4 list of 11 conversion rates, each
    one representing the level of the demand curve for a price.
    For the stationary and ab case it is possible to choose a
    row and a column. While for the nonstationary and contextual
    case it is possible to choose only the column.
    There are also some methods to get a set of conversion rates
    at random.

    Attributes
    ----------
    - n_phases (int): the number of phases
    - n_contexts (int): the number of contexts
    - prices (list): list of prices, one for each conversion rate point
    - cnv_rates (list): array 4x4 of list of 11 conversion rates
    - cnv_random_getter (dict): store the corrects methods to get
                                the conversion rates at random

    Methods
    -------
    - get_cnv_s(col=0, row=0):
        Returns the conversion rates for the stationary case
    - get_cnv_ns(col=0, row=0):
        Returns the conversion rates for the non stationary case
    - get_cnv_ctx(col=0, row=0):
        Returns the conversion rates for the contextual case
    - get_cnv_ns_ctx(col=0, row=0):
        Returns the conversion rates for the contextual non
        stationary case
    - get_random_cnv(alg_type):
        Returns a random set of conversion rates for the Bandit type
        in input
    - get_random_cnv_s():
        Returns a set of random conversion rates for the stationary
        case
    - get_random_cnv_ns():
        Returns a set of random conversion rates for the non
        stationary case
    - get_random_cnv_ctx():
        Returns a set of random conversion rates for the contextual
        case
    - parse_alg(alg):
        Returns the string identifying the correct Bandit type for
        which the conversion rates are needed
    - load_cnv_rates()
        Returns the available conversion rates
    """
    def __init__(self, n_phases=4, n_contexts=3):
        """
        Args:
            n_phases (int): the number of phases
            n_contexts (int): the number of contexts
        """
        self.n_phases = n_phases
        self.n_contexts = n_contexts
        self.prices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.cnv_rates = self.load_cnv_rates()
        self.cnv_random_getter = {
            's': self.get_random_cnv_s,
            'ns': self.get_random_cnv_ns,
            'ctx': self.get_random_cnv_ctx,
            'ctx_ns': self.get_cnv_ns_ctx}

    def get_cnv_s(self, col, row):
        """
        Args:
            col (int): the columns for the selected conversion rates
            row (int): the rows for the selected conversion rates

        Returns: The conversion rates for the stationary case.
        """
        print('Random cnv: random col = ' +
              str(col) + ' random row = ' + str(row))
        print(self.cnv_rates[col][row])

        return self.cnv_rates[col][row], self.prices


    def get_cnv_ns(self, col, row=0):
        """
        Args:
            col (int): the columns for the selected conversion rates
            row (int): the rows for the selected conversion rates

        Returns: The conversion rates for the non stationary case.
        """
        print('Random cnv: col=' + str(col))
        print(self.cnv_rates[col])

        return self.cnv_rates[col], self.prices

    def get_cnv_ctx(self, col, row=-1):
        """
        Args:
            col (int): the columns for the selected conversion rates
            row (int): the rows for the selected conversion rates

        Returns: The conversion rates for the contextual case.
        """
        print('Disaggregate cnv:')
        for c, r in zip(col, row):
            print(self.cnv_rates[c][r])
        print()
        dis_cnv = [self.cnv_rates[c][r] for c, r in zip(col, row)]
        agg_cnv = np.mean(dis_cnv, axis=0)

        return np.concatenate(([agg_cnv], dis_cnv)),\
               self.prices

    def get_cnv_ns_ctx(self, col=0, row=0):
        """
        Args:
            col (int): the columns for the selected conversion rates
            row (int): the rows for the selected conversion rates

        Returns: The conversion rates for the contextual non stationary
                 case.
        """
        dis_cnv = self.cnv_rates
        agg_cnv = [np.zeros(10) for _ in range(self.n_phases)]

        for i in range(self.n_phases):
            agg_cnv[i] = np.mean([d[i] for d in dis_cnv], axis=0)

        print("Disaggregate cnv:")
        [print(cnv, end='\n\n') for cnv in dis_cnv]
        print("Aggregate cnv:")
        [print(cnv) for cnv in agg_cnv]

        dis_cnv = functools.reduce(operator.iconcat, dis_cnv, [])

        return agg_cnv + dis_cnv

    def get_random_cnv(self, alg_type):
        """
        Args:
            alg_type (str): type of the algorithm for which we want the
                            conversion rates
        Returns: A random of conversion rates.
        """
        alg = self.parse_alg(alg_type)

        if alg is None:
            raise Exception("Algorithm keywork wrong")

        return self.cnv_random_getter[alg](), self.prices

    def get_random_cnv_s(self):
        """
        Returns: One random lists of conversion rates.
        """
        col = random.randint(0, self.n_contexts - 1)
        row = random.randint(0, self.n_phases - 1)

        print('Random cnv: random col = ' +
              str(col) + ' random row = ' + str(row))
        print(self.cnv_rates[col][row])

        return self.cnv_rates[col][row]

    def get_random_cnv_ns(self):
        """
        Returns: n random lists of conversion rates, one per phase.
        """
        col = random.randint(0, self.n_contexts - 1)

        print('Random cnv: col=' + str(col))
        print(self.cnv_rates[col])

        return self.cnv_rates[col]

    def get_random_cnv_ctx(self):
        """
        Returns: n random lists of conversion rates, one per context,
                 plus the aggregates conversion rates.
        """
        row = random.randint(0, self.n_phases - 1)

        dis_cnv = [cnv[row] for cnv in self.cnv_rates]
        agg_cnv = np.mean(dis_cnv, axis=0)

        print("Disaggregate cnv:")
        for cnv in dis_cnv:
            print(cnv)
        print("Aggregate cnv:")
        print(agg_cnv)

        return np.concatenate(([agg_cnv], dis_cnv))

    def parse_alg(self, alg):
        """
        Args:
            alg (str): type of Bandit for which we need the data

        Returns: The correct type of the Bandit
        """
        return 's' \
            if re.match('^(s.*|ab)$', alg) \
          else 'ns' \
            if re.match('^ns.*$', alg) \
          else 'ctx' \
            if re.match('^ctx(|.{4})$', alg) \
          else 'ctx_ns' \
            if re.match('^ctx_ns.*$', alg) \
          else None

    def load_cnv_rates(self):
        """
        Returns: All the conversion rates available
        """
        dis_cnv_rates = [np.zeros((self.n_phases, 11))
                         for _ in range(self.n_contexts)]

        # Context 1 Rho Fiera
        dis_cnv_rates[0][0] = np.array([1, .80, .65, .54, .43, .38, .21, .07, .01, .005, 0])  # Winter
        dis_cnv_rates[0][1] = np.array([1, .95, .90, .80, .70, .60, .45, .40, .20, .10, 0])   # Spring
        dis_cnv_rates[0][2] = np.array([1, .95, .90, .80, .68, .50, .30, .10, .06, .03, 0])   # Summer
        dis_cnv_rates[0][3] = np.array([1, .72, .54, .47, .42, .40, .35, .31, .26, .12, 0])   # Autumn

        # Context 2 Porta Garibaldi
        dis_cnv_rates[1][0] = np.array([1, .86, .78, .75, .64, .53, .32, .20, .10, .05, 0])   # Winter
        dis_cnv_rates[1][1] = np.array([1, .95, .90, .76, .61, .52, .45, .30, .15, .10, 0])   # Spring
        dis_cnv_rates[1][2] = np.array([1, .99, .92, .86, .72, .66, .54, .42, .20, .10, 0])   # Summer
        dis_cnv_rates[1][3] = np.array([1, .60, .54, .46, .40, .32, .31, .24, .12, .05, 0])   # Autumn

        # Context 3 Città Studi
        dis_cnv_rates[2][0] = np.array([1, .80, .70, .60, .50, .40, .30, .20, .10, .005, 0])  # Winter
        dis_cnv_rates[2][1] = np.array([1, .85, .80, .66, .40, .37, .31, .26, .12, .08, 0])   # Spring
        dis_cnv_rates[2][2] = np.array([1, .50, .35, .22, .15, .07, .02, .009, .005, .001, 0])# Summer
        dis_cnv_rates[2][3] = np.array([1, .85, .80, .66, .40, .37, .31, .26, .12, .08, 0])   # Autumn

        return dis_cnv_rates
