# Filename : Main_Stationary.py
# Date : 2019/05/08 12.59
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Bandits import *
from Data_Provider import *


"""
Main class of the stationary algorithms.

"""

#####################################################
#                                                   #
#   Algorithms types                                #
#                                                   #
#   - stationary: s, s_ucb, s_ths                   #
#   - nonstation: ns, ns_ucb, ns_ths                #
#   - contextual: ctx, ctx_ucb, ctx_ths             #
#   - ctxnonstat: ctx_ns, ctx_ns_ucb, ctx_ns_ths    #
#                                                   #
#####################################################

alg_type = 's'

n_experiments = 100                             # number of experiments
n_arms = 8                                      # number of arms
th = 3000                                       # time horizon

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

data_provider = Data_Provider()
cnv_rates, prices = data_provider.get_cnv_s(0, 0)

main = Bandits(n_arms, prices, cnv_rates, th, alg_type)
main.run_bandits(n_experiments)

main.plot_results()

