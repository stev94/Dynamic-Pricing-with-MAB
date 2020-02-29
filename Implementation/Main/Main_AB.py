# Filename : Main_AB.py
# Date : 2019/10/24 17.15
# Project: Dynamic Pricing
# Author : Stefano Valladares

from Bandit_algs.Bandits import *
from Data_Provider import *

alg_type = 'ab'

n_experiments = 100     # number of experiments
n_arms = 8               # number of arms
th = 3000                # time horizon

print('\n' + '#' * 80)
print("Running " + alg_type + "\n")

data_provider = Data_Provider()
cnv_rates, prices = data_provider.get_cnv_s(0,0)

main = Bandits(n_arms, prices, cnv_rates, th, alg_type)
main.run_bandits(n_experiments)

main.plot_results()
