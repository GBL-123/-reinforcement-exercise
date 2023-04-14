import numpy as np
import matplotlib.pyplot as plt
from my_class.brandit import BernoulliBrandit, ThompsonSampling
np.random.seed(666)


K = 10
T = 10000
brandit = BernoulliBrandit(K)
thompson_sampling_solver = ThompsonSampling(brandit)
thompson_sampling_solver.run(T)


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].plot(np.arange(T), thompson_sampling_solver.regrets)
ax[0].set_ylabel("accumulative regrets")
ax[0].set_xlabel("t")
ax[1].plot(np.arange(T), thompson_sampling_solver.averge_rewards)
ax[1].set_ylabel("average accumulative rewards")
ax[1].set_xlabel("t")
plt.show()












