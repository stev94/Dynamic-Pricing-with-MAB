# Filename : Main.py
# Date : 2019/10/24 03.38
# Project: Dynamic Pricing
# Author : Stefano Valladares


from Bandit_algs.Bandits import *
from Data_Provider import *
import timeit


#####################################################
#                                                   #
#   Algorithms types                                #
#                                                   #
#   - ab testing: ab                                #
#   - stationary: s, s_ucb, s_ths                   #
#   - nonstation: ns, ns_ucb, ns_ths                #
#   - contextual: ctx, ctx_ucb, ctx_ths             #
#   - ctxnonstat: ctx_ns, ctx_ns_ucb, ctx_ns_ths    #
#                                                   #
#####################################################


algs = ['ab', 's', 'ns_ths', 'ns_ucb', 'ctx_ths',
        'ctx_ucb', 'ctx_ns_ths', 'ctx_ns_ucb']

n_experiments = 500  # number of experiments
n_arms = 8           # number of arms
th = 6000            # time horizon

n_phases = 4         # number of phases
window_size = 1500   # window size

n_contexts = 3       # number of contexts

data_provider = Data_Provider(n_phases, n_contexts)

start = timeit.default_timer()

for a in algs:

    n_phases = 1 if not re.match('^.*ns.*$', a) else 4
    n_contexts = 1 if not re.match('^.*ctx.*$', a) else 3

    print('#' * 80)
    print("Running " + a + ' with params:')
    print('\t' * 5 + '-n_arms = ' + str(n_arms))
    print('\t' * 5 + '-n_exp = ' + str(n_experiments))
    print('\t' * 5 + '-time horizon = ' + str(th))
    if re.match('^.*ns.*$', a):
        print('\t' * 5 + '-n_phases = ' + str(n_phases))
        print('\t' * 5 + '-window size = ' + str(window_size))
    if re.match('^.*ctx.*$', a):
        print('\t' * 5 + '-n_contexts = ' + str(n_contexts))

    print('-' * 37 + 'INPUT' + '-' * 37)

    cnv_rates, prices = data_provider.get_random_cnv(a)

    main = Bandits(n_arms, prices, cnv_rates, th, a,
                   n_contexts=n_contexts, n_phases=n_phases,
                   ws=window_size)

    main.run_bandits(n_experiments)

    main.plot_results()

print("Total time: " + str(timeit.default_timer() - start))
