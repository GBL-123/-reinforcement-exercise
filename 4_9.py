import numpy as np
import matplotlib.pyplot as plt


class GambleProblem():
    '''赌徒问题'''
    
    def __init__(self, prob, goal, gamma):
        '''
        prob: 硬币正面的概率
        goal: 目标
        gamma: 折扣因子
        '''
        self.prob = prob
        self.goal = goal
        self.gamma = gamma
        self.vs = np.zeros(self.goal + 1) # 初始V(s)全为0，为方便加了状态0和goal
        
        
    def value_iteration(self):
        '''价值迭代算法'''
        
        count = 0
        while True:
            v = self.vs.copy()
            for s in range(1, self.goal): # 遍历状态空间
                qsa_list = [] # 记录Q(s,a)的值
                for a in range(1, min(s, self.goal - s) + 1): # 遍历动作空间
                    qsa = self.__calculate_qsa(s, a)
                    qsa_list.append(qsa)
                max_qsa = np.max(qsa_list)
                self.vs[s] = max_qsa
            delta = np.max(np.abs(v - self.vs))
            count += 1

            if delta < 1e-6: # 价值迭代结束条件
                break
        print("价值迭代一共进行{}轮。".format(count))
        
        self.pi = self.__get_policy() # 获得策略
        self.vs = self.vs[1:-1] # 去掉0和goal对应的V(s)
        
    
    def __calculate_qsa(self, s, a):
        '''
        计算Q(s,a)

        s: 当前状态
        a: 选择的动作
        '''

        p = np.array([self.prob, 1 - self.prob]) # 转移概率
        next_state = np.array([s + a, s - a]) # 下一步可能转移到的状态

        # 根据下一个状态定义不同的reward
        if next_state[0] == self.goal:
            reward = np.array([1, 0])
            done = np.array([1, 0])
        elif next_state[1] == 0:
            reward = np.array([0, 0])
            done = np.array([0, 1])
        else:
            reward = np.array([0, 0])
            done = np.array([0, 0])

        qsa = np.sum(p * (reward + self.gamma * self.vs[next_state] * (1 - done)))

        return qsa
    
    
    def __get_policy(self):
        '''获得价值迭代的策略'''
        
        pi = np.zeros(self.goal + 1) # 初始策略全为0
        for s in range(1, self.goal):
            qsa_list = []
            for a in range(1, min(s, self.goal - s) + 1):
                qsa = self.__calculate_qsa(s, a)
                qsa_list.append(qsa)
            best_a = np.argmax(np.round(qsa_list, 5)) + 1
            pi[s] = best_a
        
        return pi[1:-1] # 去掉0和goal对应的策略
    

gamble_problem = GambleProblem(prob=0.55, goal=100, gamma=1)
gamble_problem.value_iteration()

fig, ax = plt.subplots(1, 2, figsize=(14,6))
ax[0].bar(np.arange(1, 100), gamble_problem.pi)
ax[0].set_xlabel("capital")
ax[0].set_ylabel("policy")
ax[0].set_title("best policy")
ax[1].plot(np.arange(1, 100), gamble_problem.vs)
ax[1].set_xlabel("capital")
ax[1].set_ylabel("V(s)")
ax[1].set_title("max V(s)")
plt.show()