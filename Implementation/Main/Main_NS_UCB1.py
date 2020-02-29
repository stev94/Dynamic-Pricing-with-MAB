# Filename : Main_SWUCB1.py
# Date : 2019/05/09 03.22
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Bandits import *
from Data_Provider import *

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

"""
Main class of the NS-UCB1 algorithm.

"""
alg_type = 'ns_ucb'

n_experiments = 1               # number of experiments
n_arms = 8                      # number of arms
th = 6000                        # time horizon

n_phases = 4                    # number of phases
window_size = 1500               # window size

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

data_provider = Data_Provider(n_phases)
cnv_rates, prices = data_provider.get_random_cnv(alg_type)

main = Bandits(n_arms, prices, cnv_rates, th, alg_type,
               n_phases=n_phases, ws=window_size)

main.run_bandits(n_experiments)

main.plot_results()
