from Bandit_algs.Bandit import *
from scipy import stats
import random
import numpy as np


class Sequential_AB(Bandit):

    def __init__(self, th, n_arms, prices):
        self.n_samples = 400       # Number of samples drawn when comparing 2 arms
        self.accuracy = .001       # Accuracy of the test. Used for comparing the final values of the 2 tested arms
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        super().__init__(n_arms, prices, 'ab', th)

    def execute(self, env):

        collected_reward = []
        arms = [i for i in range(self.n_arms)]
        random.shuffle(arms)

        for t in range(0, self.time_horizon, self.n_samples):
            samples = [[] for _ in range(2)]

            for _ in range(self.n_samples):
                reward, pull = self.observe_sample(arms, env, collected_reward)
                samples[pull].append(reward * self.prices[arms[pull]])

            if len(arms) > 1:
                kstat, p_val = stats.ks_2samp(samples[0], samples[1])
                if self.accuracy > p_val:
                    arms.remove(arms[np.mean(samples[0]) > np.mean(samples[1])])
                else:
                    random.shuffle(arms)

        # Se il th è abbstanza grande non serve
        for t in range(int(self.time_horizon / self.n_samples) *
                       self.n_samples, self.time_horizon):
            self.observe_sample(arms, env, collected_reward)

        self.rewards_per_experiment.append(collected_reward)
        for i in range(self.n_arms):
            self.rewards_per_exp_per_arm[i].append(np.mean(self.rewards_per_arm[i]))

    def observe_sample(self, arms, env, collected_reward):
        # Random number, uniformely drawn, to decide which arm to pull
        pull = np.random.binomial(1, 0.5) if len(arms) > 1 else 0

        reward = env.round(arms[pull])
        self.rewards_per_arm[arms[pull]].append(reward)
        collected_reward.append(reward * self.prices[arms[pull]])

        return reward, pull

    def get_arms_cb_estimates(self):
        return np.array([0 for _ in range(len(np.mean(self.rewards_per_exp_per_arm, axis=1)))])
