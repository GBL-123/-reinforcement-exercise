import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


class WindyGridWorld():
    '''有风的网格世界'''
    def __init__(self, nrows, ncols, alpha, gamma, num_episode, epsilon):
        '''
        nrows: 行数
        ncols: 列数
        alpha: 学习率
        gamma: 折扣因子
        num_episode: 学习的幕数
        epsilon: epsilon greedy的参数
        '''
        self.nrows = nrows
        self.ncols = ncols
        self.alpha = alpha
        self.gamma = gamma
        self.num_episode = num_episode
        self.actions = np.arange(4) # 能选择的动作, 0-3分别表示上下左右
        self.action_marks = ["↑", "↓", "←", "→"] # 动作标记
        self.qsa = np.zeros((self.nrows, self.ncols, len(self.actions))) # 初始化Q(s,a)为0，shape(行数，列数，动作数)
        self.wind1 = [3, 4, 5, 8] # 有1级风的列，列的index从0开始
        self.wind2 = [6, 7] # 有2级风的列，列的index从0开始
        self.epsilon = epsilon
        self.pi = np.zeros((self.nrows, self.ncols)) # 初始策略
        self.start_i, self.start_j = 3, 0 # 初始位置
        self.route_i = [self.start_i] # 记录智能体的行坐标
        self.route_j = [self.start_j] # 记录智能体的列坐标
        self.end_i, self.end_j = 3, 7 # 终止位置
        self.time = [] # 每一幕需要的步数

    
    def _take_action(self, state_i, state_j, t):
        '''使用decay-epsilon贪心算法选择动作'''
        if np.random.rand() < self.epsilon / (t + 1):
            action = np.random.randint(len(self.actions))
        else:
            action = np.argmax(self.qsa[state_i, state_j])

        return action
    

    def _get_next_state(self, state_i, state_j, action):
        '''
        给定当前状态，执行动作后转移到的下一个状态

        state_i: 当前行状态
        state_j: 当前列状态
        action: 选择的动作
        '''
        # 正常走
        if action == 0:
            next_state_i, next_state_j = max(0, state_i - 1), state_j
        elif action == 1:
            next_state_i, next_state_j = min(self.nrows - 1, state_i + 1), state_j
        elif action == 2:
            next_state_i, next_state_j = state_i, max(0, state_j - 1)
        else:
            next_state_i, next_state_j = state_i, min(self.ncols - 1, state_j + 1)
        
        # 当前状态在有风的地方
        if state_j in self.wind1: # 1级风
            next_state_i = max(next_state_i - 1, 0)
        if state_j in self.wind2: # 2级风
            next_state_i = max(next_state_i - 2, 0)

        return next_state_i, next_state_j


    def sarsa(self):
        '''执行sarsa算法'''
        for t in range(self.num_episode):
            state_i, state_j = self.start_i, self.start_j # 初始状态
            action = self._take_action(state_i, state_j, t)

            # 对每一幕的循环
            count = 0 # 记录这一幕要走多少步
            route_i, route_j = [], [] # 记录这一幕的路线
            while True: 
                count += 1
                next_state_i, next_state_j = self._get_next_state(state_i, state_j, action)
                route_i.append(next_state_i)
                route_j.append(next_state_j)
                
                # 下一个状态到终点就结束这一幕
                if next_state_i == self.end_i and next_state_j == self.end_j: 
                    break
                
                reward = -1
                next_action = self._take_action(next_state_i, next_state_j, t)
                self.qsa[state_i, state_j, action] += self.alpha * (reward + \
                    self.gamma * self.qsa[next_state_i, next_state_j, next_action] - \
                    self.qsa[state_i, state_j, action])
                state_i, state_j, action = next_state_i, next_state_j, next_action
                
            self.time.append(count)

        self._get_policy()
        self.route_i.extend(route_i)
        self.route_j.extend(route_j)
    

    def _get_policy(self):
        '''获得最佳策略'''
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.pi[i, j] = np.argmax(self.qsa[i, j])
                
    
    def print_policy(self):
        '''打印最佳策略'''
        for i in range(self.nrows):
            for j in range(self.ncols):
                print(self.action_marks[int(self.pi[i, j])], end=" ")
            print()
        print()


    def plot_route(self, ax):
        '''画路线图'''
        ax.plot([self.start_j] + self.route_j, [self.start_i] + self.route_i)
        ax.set_ylim(-0.5, self.nrows - 0.5)
        ax.set_xlim(-0.5, self.ncols - 0.5)
        ax.hlines(np.arange(-0.5, self.nrows + 1), -1, self.ncols + 1, colors="black", lw=0.5)
        ax.vlines(np.arange(-0.5, self.ncols + 1), -1, self.nrows + 1, colors="black", lw=0.5)
        ax.text(self.start_j, self.start_i, "S", size=20, ha="center", va="center")
        ax.text(self.end_j, self.end_i, "G", size=20, ha="center", va="center")
        ax.axvspan(2.5, 5.5, color="orange", alpha=0.2) # 1级风的区域
        ax.axvspan(7.5, 8.5, color="orange", alpha=0.2) # 1级风的区域
        ax.axvspan(5.5, 7.5, color="red", alpha=0.2) # 2级风的区域
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    
    def plot_use_time(self, ax):
        '''画每一幕的耗时'''
        ax.plot(np.arange(1, self.num_episode + 1), self.time)
        ax.set_xlabel("episode numbers")
        ax.set_ylabel("use time")

    
    def plot(self):
        '''画路线图和每一幕的耗时'''
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        self.plot_route(ax=ax[0])
        self.plot_use_time(ax=ax[1])
        plt.show()
