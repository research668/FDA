import numpy as np
from itertools import product
from scipy.spatial.distance import cdist
import multiprocessing

# def gaussian_kernel(x, bandwidth):
#     u = x / bandwidth
#     return np.exp(-u**2/2) / (bandwidth * np.sqrt(2*np.pi))

# function: $w_i = \frac{1}{\sum_{j=1}^{n} \exp\left(\frac{\|x_{n+1} - x_i\|^2 - \|x_{n+1} - x_j\|^2}{2w^2}\right)}$.
def gaussian_kernel(x, w):
    return np.exp(x / (2 * w**2))

class KernelOptimization:
    def __init__(self, backorder_cost, holding_cost):
        self.b = backorder_cost
        self.h = holding_cost
    
    def kernel_optimization(self, X_hat, d_hat, X_test, w):
        # Calculate the distance from x_test to all elements in X_hat
        distances = cdist(X_test, X_hat, metric='euclidean')

        # Calculate the distance differences
        distance_sq = distances ** 2
        distance_sq = distance_sq.astype(np.float32)

        exponent = distance_sq / (2 * w**2)
        
        # 数值稳定性处理 (log-sum-exp技巧)
        min_exponent = np.min(exponent, axis=1, keepdims=True)
        exp_terms = np.exp(exponent - min_exponent)  # 稳定化后的指数项

        # 计算归一化权重 (n_test, n_train)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True) + 1e-12
        weights = exp_terms / sum_exp
        
        # 排序并计算累积权重
        sorted_indices = np.argsort(d_hat)
        sorted_weights = weights[:, sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights, axis=1)
        
        # 确定分位点
        threshold = self.b / (self.b + self.h)
        mask = cumulative_weights >= threshold
        indices = np.argmax(mask, axis=1)
        
        # 处理边界条件
        valid = np.any(mask, axis=1)

        q_values = np.where(
            valid,
            d_hat[sorted_indices[indices]],
            d_hat[sorted_indices[-1]]  # 默认取最大值
        )
        

        # print("\n d_hat: ", d_hat)
        # print("\n weights: ", weights)
        # print("\n sort: ", d_hat[sorted_indices])
        # print("\n d_cumulative_weights: ", cumulative_weights)
        # print("\n sorted_indices: ", sorted_indices)
        # print("\n threshold: ", threshold)
        # print("\n idx: ", indices)   
        # print("\n q: ", d_hat[sorted_indices][indices])     

        return d_hat[sorted_indices][indices]

    def find_optimal_bandwidth(self, k_fold_sets, b_min, b_max, n_points=30):

        # 生成对数均匀分布的候选带宽
        bandwidths = np.logspace(
            np.log10(b_min),
            np.log10(b_max),
            num=n_points
        )

        min_cost, optimal_bandwidth = float('inf'), b_min
        tasks = []

        for w in bandwidths:
            tasks.append((w,k_fold_sets))

        pool = multiprocessing.Pool(processes = 30)
        results = pool.starmap(self.calculate_ko_cost,tasks)
        w_dict = {}
        for idx, cost in enumerate(results):
            w_dict[bandwidths[idx]] = cost
        pool.close()
        optimal_bandwidth =  min(w_dict, key=w_dict.get)
            # c = self.calculate_ko_cost(w, k_fold_sets)
            # if c < min_cost:
            #     min_cost, optimal_bandwidth = c, w

        return optimal_bandwidth
    
    def calculate_ko_cost(self, w, k_fold_sets):
        """
        计算核优化（KO）的成本。
        """
        total_cost = 0.0

        for (X_train, d_train), (X_val, d_val) in k_fold_sets:

            # 计算 q 值
            q_values = self.kernel_optimization(X_train, d_train, X_val, w)

            # 计算新闻供应链成本
            cost = self.newsvendor_cost(q_values, d_val)
            total_cost += np.sum(cost)

        return total_cost
    
    def newsvendor_cost(self, q, d):
        """计算报童问题成本"""
        tau = self.b / (self.h + self.b)
        return (1-tau) * np.maximum(q - d, 0) + tau * np.maximum(d - q, 0)

    def _find_optimal_s_l(self, k_fold_sets, s_values, l_values, optimal_w):
        """通过交叉验证寻找最优 beta 和 rho"""
        min_cost = float('inf')
        optimal_s, optimal_l = s_values[0], l_values[0]
        
        # 网格搜索所有参数组合
        for s, l in product(s_values, l_values):
            cost = self._calculate_boosting_cost(s, l, k_fold_sets, optimal_w)
            if cost < min_cost:
                min_cost = cost
                optimal_s, optimal_l = s, l
                
        return optimal_s, optimal_l


    def _calculate_boosting_cost(self, s, l, k_fold_sets, optimal_w):
        """
        计算核优化（KO）的成本。
        """
        total_cost = 0.0

        for (X_train, d_train), (X_val, d_val) in k_fold_sets:

            # 计算 q 值
            q_values = [self.kernel_optimization(X_train, d_train, X_val[i, :], optimal_w) for i in range(len(d_val))]

            q_optimal = np.array(q_values)
            q_boosting = s*q_optimal + l

            # 计算新闻供应链成本
            cost = self.newsvendor_cost(q_boosting, d_val)
            total_cost += np.sum(cost)

        return total_cost

'''
Example:

import numpy as np
from scipy.spatial.distance import cdist

def gaussian_kernel(x, w):
    return np.exp(x / (2 * w**2))

class KernelOptimization:
    def __init__(self, backorder_cost, holding_cost):
        self.b = backorder_cost
        self.h = holding_cost

    def kernel_optimization(self, X_hat, d_hat, x_test, w):
        # Calculate the distance from x_test to all elements in X_hat
        distances = cdist([x_test], X_hat, metric='euclidean').flatten()
        print("Distances:", distances)
        
        # Calculate the distance differences
        distance_sq = distances ** 2
        distance_diff = distance_sq[:, None] - distance_sq[None, :]
        print("Distance differences:", distance_diff)
        
        # Calculate the kernel weights
        kernel_matrix = gaussian_kernel(distance_diff, w)
        weights = 1 / kernel_matrix.sum(axis=1)
        print("kernel_matrix:", kernel_matrix)
        print("Weights:", weights)
        
        sorted_indices = np.argsort(d_hat)
        print("Sorted indices:", sorted_indices)
        
        cumulative_weights = np.cumsum(weights[sorted_indices])
        print("Cumulative weights:", cumulative_weights)

        threshold = self.b / (self.b + self.h)
        indices = np.where(cumulative_weights >= threshold)[0]
        idx = indices[0] if indices.size > 0 else -1

        return d_hat[idx]

# Example usage
if __name__ == "__main__":
    X_hat = np.array([[1, 2], [3, 4], [5, 6]])
    d_hat = np.array([20, 10, 30])
    x_test = np.array([2, 3])
    w = 1.0

    ko = KernelOptimization(backorder_cost=1, holding_cost=1)
    result = ko.kernel_optimization(X_hat, d_hat, x_test, w)
    print("Optimized value:", result)

Output:
Distances: [1.41421356 1.41421356 4.24264069]
Distance differences: [[  0.   0. -16.]
 [  0.   0. -16.]
 [ 16.  16.   0.]]
kernel_matrix: [[1.00000000e+00 1.00000000e+00 3.35462628e-04]
 [1.00000000e+00 1.00000000e+00 3.35462628e-04]
 [2.98095799e+03 2.98095799e+03 1.00000000e+00]]
Weights: [4.99916148e-01 4.99916148e-01 1.67703185e-04]
Sorted indices: [1 0 2]
Cumulative weights: [0.49991615 0.9998323  1.        ]
Optimized value: 10

'''