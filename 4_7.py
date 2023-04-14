import numpy as np
from scipy import stats
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns


@jit
def _calculate_qsa(
    s1, s2, a, 
    prob_rent1, prob_rent2, 
    prob_return1, prob_return2, 
    max_car1, max_car2, 
    vs, rent_income, move_cost, gamma, 
):
    '''
    计算Q(s,a)
    
    s1: 1地当前状态
    s2: 2地当前状态
    prob_rent1: 1地租车概率
    prob_rent2: 2地租车概率
    prob_return1: 1地还车概率
    prob_return2: 2地还车概率
    max_car1: 1地最大车辆
    max_car2: 2地最大车辆
    vs: 当前价值函数
    rent_income: 租车收益
    move_cost: 移车花费
    gamma: 折扣因子
    '''

    qsa = 0
    s1_after_a = min(s1 + a, max_car1) # 移车后1地状态
    s2_after_a = min(s2 - a, max_car2) # 移车后2地状态
    
    fixed_cost = 0
    # 计算固定支出
    if a < 0 :
        fixed_cost += (np.abs(a) - 1) * move_cost
    else:
        fixed_cost += np.abs(a) * move_cost
    
    if s1_after_a > 10 or s2_after_a > 10:
        fixed_cost += 4

    for rent1 in range(max_car1 + 1): # 1地租车
        for rent2 in range(max_car2 + 1): # 2地租车
            
            # 实际能租出去的车
            real_rent1 = min(s1_after_a, rent1)
            real_rent2 = min(s2_after_a, rent2)
            
            # 计算收益
            reward = rent_income * (real_rent1 + real_rent2) - fixed_cost
            
            # 租完剩下的车
            s1_after_rent = s1_after_a - real_rent1
            s2_after_rent = s2_after_a - real_rent2
            
            for return1 in range(max_car1 + 1): # 1地还车
                for return2 in range(max_car2 + 1): # 2地还车
                    
                    # 计算转移概率
                    p = prob_rent1[rent1] * prob_rent2[rent2] * prob_return1[return1] * prob_return2[return2]

                    # 转移到的状态
                    next_s1 = min(s1_after_rent + return1, max_car1)
                    next_s2 = min(s2_after_rent + return2, max_car2)
                    
                    qsa += p * (reward + gamma * vs[int(next_s1), int(next_s2)])

    return qsa
    

class RentalProblem():
    '''租车问题'''
    
    def __init__(
        self, 
        max_car1, max_car2, 
        rent_mu1, rent_mu2, 
        return_mu1, return_mu2, 
        rent_income, move_cost, 
        gamma
    ):
        '''
        max_car1: 1地的最大车辆
        max_car2: 2地的最大车辆
        rent_mu1: 1地租车泊松均值
        rent_mu2: 2地租车泊松均值
        return_mu1: 1地租车泊松均值
        return_mu2: 2地租车泊松均值
        rent_income: 租一辆车的收益
        move_cost: 移动一辆车的花费
        gamma: 折扣因子
        '''
        
        self.max_car1 = max_car1
        self.max_car2 = max_car2
        self.rent_mu1 = rent_mu1
        self.rent_mu2 = rent_mu2
        self.return_mu1 = return_mu1
        self.return_mu2 = return_mu2
        self.rent_income = rent_income
        self.move_cost = move_cost
        self.gamma = gamma
        self.vs = np.zeros((max_car1 + 1, max_car2 + 1)) # 初始化V(s)全为0，是一个矩阵，行代表1地状态，列代表2地状态
        self.pi = np.zeros((max_car1 + 1, max_car2 + 1)) # 初始化策略
        
        # 计算借车还车的概率
        self.__prob_rent1 = stats.poisson.pmf(np.arange(self.max_car1 + 1), self.rent_mu1)
        self.__prob_rent1[-1] = 1 - stats.poisson.cdf(self.max_car1 - 1, self.rent_mu1)
        self.__prob_rent2 = stats.poisson.pmf(np.arange(self.max_car2 + 1), self.rent_mu2)
        self.__prob_rent2[-1] = 1 - stats.poisson.cdf(self.max_car2 - 1, self.rent_mu2)
        self.__prob_return1 = stats.poisson.pmf(np.arange(self.max_car1 + 1), self.return_mu1)
        self.__prob_return1[-1] = 1 - stats.poisson.cdf(self.max_car1 - 1, self.return_mu1)
        self.__prob_return2 = stats.poisson.pmf(np.arange(self.max_car2 + 1), self.return_mu2)
        self.__prob_return2[-1] = 1 - stats.poisson.cdf(self.max_car2 - 1, self.return_mu2)
    
    
    def policy_iteration(self):
        '''策略迭代算法'''
        
        count = 0
        while True:
            # 策略评估
            self.__policy_evaluation()

            # 策略提升
            old_pi = self.pi.copy()
            self.__policy_improvement()
            count += 1
            if (old_pi == self.pi).all(): # 策略迭代结束
                break
        print("共进行了{}次策略迭代。".format(count))
    
    
    def __policy_evaluation(self):
        '''策略评估'''
    
        count = 0
        while True:
            v = self.vs.copy()
            for s1 in range(self.max_car1 + 1):
                for s2 in range(self.max_car2 + 1): # 遍历所有状态空间
                    self.vs[s1, s2] = self.__calculate_qsa(s1, s2, self.pi[s1, s2]) # 当前策略下的V(s)
            delta = np.max(np.abs(v - self.vs))

            count += 1
            if delta < 1e-6: # 迭代结束条件
                break
        print("策略评估一共进行{}轮。".format(count))
    
    
    def __calculate_qsa(self, s1, s2, a):
        '''计算Q(s,a)'''
        
        qsa = _calculate_qsa(
            s1, s2, a, 
            self.__prob_rent1, self.__prob_rent2, 
            self.__prob_return1, self.__prob_return2, 
            self.max_car1, self.max_car2, 
            self.vs, self.rent_income, self.move_cost, self.gamma
        )

        return qsa
    
    
    def __policy_improvement(self):
        '''策略提升'''

        for s1 in range(self.max_car1 + 1):
            for s2 in range(self.max_car2 + 1): # 遍历所有状态空间
                qsa_list = [] # 记录Q(s,a)的值
                actions = np.arange(-min(s1, 5), min(s2, 5) + 1) # 可能的动作
                for a in actions: # 遍历动作空间
                    qsa = self.__calculate_qsa(s1, s2, a)
                    qsa_list.append(qsa)
                self.pi[s1, s2] = actions[np.argmax(qsa_list)]
        print("策略提升完成。")


rental_problem = RentalProblem(
    max_car1=20, max_car2=20,
    rent_mu1=3, rent_mu2=4, 
    return_mu1=3, return_mu2=2, 
    rent_income=10, move_cost=2, 
    gamma=0.9
)
rental_problem.policy_iteration()

fig, ax = plt.subplots(1, 2, figsize=(14,5))
sns.heatmap(rental_problem.pi, annot=True, cmap="RdBu", center=0, ax=ax[0])
ax[0].set_ylabel("first location")
ax[0].set_xlabel("second location")
ax[0].set_title("best policy")
sns.heatmap(rental_problem.vs, ax=ax[1])
ax[1].set_ylabel("first location")
ax[1].set_xlabel("second location")
ax[1].set_title("max V(s)")
plt.show()
