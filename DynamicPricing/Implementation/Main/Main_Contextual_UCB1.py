# Filename : Main_Contextual_UCB1.py
# Date : 2019/10/24 01.53
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

alg_type = 'ctx_ucb'

n_experiments = 1       # number of experiments
n_arms = 9              # number of arms
th = 6000                # time horizon

n_contexts = 3          # number of contexts

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

data_provider = Data_Provider()
cnv_rates, prices = data_provider.get_random_cnv(alg_type)

main = Bandits(n_arms, prices, cnv_rates, th, alg_type, n_contexts)
main.run_bandits(n_experiments)

main.plot_results()
