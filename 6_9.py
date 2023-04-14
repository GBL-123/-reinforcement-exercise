import numpy as np

from my_class.windy_grid_world import WindyGridWorld

np.random.seed(666)


class WindyGridWorldWithEightActions(WindyGridWorld):
    '''能选择8个动作的网格世界，继承自WindyGridWorld'''
    def __init__(self, nrows, ncols, alpha, gamma, num_episode, epsilon):
        '''
        nrows: 行数
        ncols: 列数
        alpha: 学习率
        gamma: 折扣因子
        num_episode: 学习的幕数
        epsilon: epsilon greedy的参数
        '''
        super().__init__(nrows, ncols, alpha, gamma, num_episode, epsilon)
        self.actions = np.arange(8) # 8个动作，0-8分别对应上下左右，左上、右上、左下、右下
        self.action_marks = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘"]
        self.qsa = np.zeros((self.nrows, self.ncols, len(self.actions))) # 初始化Q(s,a)为0，shape(行数，列数，动作数)


    def _get_next_state(self, state_i, state_j, action):
        '''
        重写父类方法，给定当前状态，执行动作后转移到的下一个状态

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
        elif action == 3:
            next_state_i, next_state_j = state_i, min(self.ncols - 1, state_j + 1)
        elif action == 4:
            next_state_i, next_state_j = max(0, state_i - 1), max(0, state_j - 1)
        elif action == 5:
            next_state_i, next_state_j = max(0, state_i - 1), min(self.ncols - 1, state_j + 1)
        elif action == 6:
            next_state_i, next_state_j = min(self.nrows - 1, state_i + 1), max(0, state_j - 1)
        else:
            next_state_i, next_state_j = min(self.nrows - 1, state_i + 1), min(self.ncols - 1, state_j + 1)
        
        # 当前状态在有风的地方
        if state_j in self.wind1: # 1级风
            next_state_i = max(next_state_i - 1, 0)
        if state_j in self.wind2: # 2级风
            next_state_i = max(next_state_i - 2, 0)

        return next_state_i, next_state_j


class WindyGridWorldWithNineActions(WindyGridWorld):
    '''能选择9个动作的网格世界，继承自WindyGridWorld'''
    def __init__(self, nrows, ncols, alpha, gamma, num_episode, epsilon):
        '''
        nrows: 行数
        ncols: 列数
        alpha: 学习率
        gamma: 折扣因子
        num_episode: 学习的幕数
        epsilon: epsilon greedy的参数
        '''
        super().__init__(nrows, ncols, alpha, gamma, num_episode, epsilon)
        self.actions = np.arange(9) # 9个动作，0-9分别对应上下左右，左上、右上、左下、右下、不动
        self.action_marks = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "·"]
        self.qsa = np.zeros((self.nrows, self.ncols, len(self.actions))) # 初始化Q(s,a)为0，shape(行数，列数，动作数)


    def _get_next_state(self, state_i, state_j, action):
        '''
        重写父类方法，给定当前状态，执行动作后转移到的下一个状态

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
        elif action == 3:
            next_state_i, next_state_j = state_i, min(self.ncols - 1, state_j + 1)
        elif action == 4:
            next_state_i, next_state_j = max(0, state_i - 1), max(0, state_j - 1)
        elif action == 5:
            next_state_i, next_state_j = max(0, state_i - 1), min(self.ncols - 1, state_j + 1)
        elif action == 6:
            next_state_i, next_state_j = min(self.nrows - 1, state_i + 1), max(0, state_j - 1)
        elif action == 7:
            next_state_i, next_state_j = min(self.nrows - 1, state_i + 1), min(self.ncols - 1, state_j + 1)
        else:
            next_state_i, next_state_j = state_i, state_j
        
        # 当前状态在有风的地方
        if state_j in self.wind1: # 1级风
            next_state_i = max(next_state_i - 1, 0)
        if state_j in self.wind2: # 2级风
            next_state_i = max(next_state_i - 2, 0)

        return next_state_i, next_state_j
        

# 能选择8个动作的情况
windy_grid_world_with_eight_actions = WindyGridWorldWithEightActions(7, 10, 0.5, 1, 500, 1)
windy_grid_world_with_eight_actions.sarsa()
windy_grid_world_with_eight_actions.print_policy()
windy_grid_world_with_eight_actions.plot()

# 能选择9个动作的情况
windy_grid_world_with_nine_actions = WindyGridWorldWithNineActions(7, 10, 0.5, 1, 500, 1)
windy_grid_world_with_nine_actions.sarsa()
windy_grid_world_with_nine_actions.print_policy()
windy_grid_world_with_nine_actions.plot()



