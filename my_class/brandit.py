import numpy as np


class BernoulliBrandit():
    '''伯努利多臂老虎机'''

    def __init__(self, K):
        '''K为拉杆数'''
        self.probs = np.random.rand(K) # k个臂的获奖概率
        self.best_idx = np.argmax(self.probs) # 获得获奖概率最大的臂
        self.best_prob = self.probs[self.best_idx] # 获奖最大概率
        self.K = K

    def get_reward(self, k):
        '''选择第k个拉杆以后返回是否获奖，1获奖，0不获奖'''
        if np.random.binomial(n=1, p=self.probs[k]) > 0.5:
            return 1
        else:
            return 0


class BernoulliBranditWithNoise(BernoulliBrandit):
    '''奖励带随机游走的伯努利多臂老虎机'''

    def __init__(self, K, scale=0.1):
        '''K为拉杆数，scale为随机游走的标准差'''
        super().__init__(K)
        self.noise = np.random.normal(scale=scale)

    def get_reward(self, k):
        '''选择第k个拉杆以后返回是否获奖，1获奖，0不获奖，获奖后加一个随机游走'''
        if np.random.binomial(n=1, p=self.probs[k]) > 0.5:
            reward = 1
        else:
            reward = 0

        return reward + self.noise


class Solver():
    '''多臂老虎机算法基本框架'''

    def __init__(self, brandit):
        '''brandit为多臂老虎机的类'''
        self.brandit = brandit
        self.N = np.zeros(self.brandit.K) # 每个臂的拉动次数
        self.regret = 0 # 当前regret
        self.actions = [] # 记录每一次拉动的杆
        self.regrets = [] # 记录每一次的累计regret
        self.Q = np.zeros(self.brandit.K) # 每个臂的期望奖励估计值
        self.rewards = 0 # 记录累计收益
        self.averge_rewards = [] # 记录到目前为止的平均累计收益

    def update_regret(self, k):
        '''计算每一步的regret，k为当前选择的臂'''
        self.regret += self.brandit.best_prob - self.brandit.probs[k]
        self.regrets.append(self.regret)

    def update_Q(self, k, t):
        '''计算每一步的期望奖励估计值Q，k为当前选择的臂，t为当前已执行的时间步'''
        reward = self.brandit.get_reward(k)
        self.Q[k] += 1/self.N[k] * (reward - self.Q[k])
        self.rewards += reward
        self.averge_rewards.append(self.rewards/(t+1))
    
    def run_one_step(self):
        '''返回当前选择的杆，需要根据不同策略对该方法进行重写'''
        raise NotImplementedError("方法没有重写")
    
    def run(self, T):
        '''运行一定次数，T为运行的总次数'''
        for t in range(T):
            k = self.run_one_step()
            self.N[k] += 1
            self.actions.append(k)
            self.update_regret(k)
            self.update_Q(k, t)


class EpsilonGreedy(Solver):
    '''epsilon greedy算法'''

    def __init__(self, brandit, epsilon=0.1):
        '''brandit为多臂老虎机类，epsilon为算法的随机选择概率'''
        super().__init__(brandit)
        self.epsilon = epsilon

    def run_one_step(self):
        '''执行一次epsilon greedy算法'''
        if np.random.binomial(1, self.epsilon) > 0.5: 
            k = np.random.randint(0, self.brandit.K) # 随机选择 
        else:
            k = np.argmax(self.Q) # 选择当前期望奖励估计最大的臂
        
        return k


class ThompsonSampling(Solver):
    '''汤普森采样算法'''

    def __init__(self, brandit):
        '''brandit为多臂老虎机类'''
        super().__init__(brandit)
        self.alpha = np.ones(self.brandit.K) # beta分布的第一个参数
        self.beta = np.ones(self.brandit.K) # beta分布的第二个参数 

    def run_one_step(self):
        '''执行一次汤普森采样'''
        self.probs_hat = np.random.beta(self.alpha, self.beta) # 按照beta分布生成每个老虎机获得reward的概率

        k = np.argmax(self.probs_hat)
        reward = self.brandit.get_reward(k)
        self.alpha[k] += reward
        self.beta[k] += 1 - reward

        return k























