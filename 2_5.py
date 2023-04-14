import numpy as np
import matplotlib.pyplot as plt
from my_class.brandit import BernoulliBrandit, BernoulliBranditWithNoise, EpsilonGreedy
np.random.seed(666)


K = 10
T = 10000
epsilon = 0.1
brandit = BernoulliBrandit(K)
epsilon_greedy_solver = EpsilonGreedy(brandit, epsilon=epsilon)
epsilon_greedy_solver.run(T)

brandit_with_noise = BernoulliBranditWithNoise(K)
epsilon_greedy_solver2 = EpsilonGreedy(brandit, epsilon=epsilon)
epsilon_greedy_solver2.run(T)


fig, ax = plt.subplots(1, 2, figsize=(14,6))
ax[0].plot(np.arange(T), epsilon_greedy_solver.regrets, label="without noise")
ax[0].set_ylabel("accumulative regrets")
ax[0].set_xlabel("t")
ax[0].plot(np.arange(T), epsilon_greedy_solver2.regrets, label="with noise")
ax[0].set_ylabel("accumulative regrets")
ax[0].set_xlabel("t")
ax[0].legend()

ax[1].plot(np.arange(T), epsilon_greedy_solver.averge_rewards, label="without noise")
ax[1].set_ylabel("average accumulative rewards")
ax[1].set_xlabel("t")
ax[1].plot(np.arange(T), epsilon_greedy_solver2.averge_rewards, label="with noise")
ax[1].set_ylabel("average accumulative rewards")
ax[1].set_xlabel("t")
ax[1].legend()
plt.show()













