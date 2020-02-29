# Filename : Main_Non_Stationary.py
# Date : 2019/10/02 19.43
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

alg_type = 'ns'

n_experiments = 100  # number of experiments
n_arms = 10  # number of arms
th = 6000  # time horizon

n_phases = 4  # number of phases
window_size = 1500  # window size

data_provider = Data_Provider(n_phases)
cnv_rates, prices = data_provider.get_cnv_ns(0)

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

main = Bandits(n_arms, prices, cnv_rates, th, alg_type,
               n_phases=n_phases, ws=window_size)
main.run_bandits(n_experiments)

main.plot_results()
