# Filename : Contextual_Main.py
# Date : 2019/10/17 12.22
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


alg_type = 'ctx'

n_experiments = 100  # number of experiments
n_arms = 8  # number of arms
th = 6000  # time horizon

n_contexts = 3  # number of contexts

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

data_provider = Data_Provider(n_contexts=n_contexts)
cnv_rates, prices = data_provider.get_cnv_ctx([0, 1, 2], [0, 1, 2])

main = Bandits(n_arms, prices, cnv_rates, th, alg_type, n_contexts)
main.run_bandits(n_experiments)

main.plot_results()
