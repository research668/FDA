import numpy as np
from itertools import product
from scipy.spatial.distance import cdist
import multiprocessing

.
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
        
    
        min_exponent = np.min(exponent, axis=1, keepdims=True)
        exp_terms = np.exp(exponent - min_exponent)  # 稳定化后的指数项

   
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True) + 1e-12
        weights = exp_terms / sum_exp
        

        sorted_indices = np.argsort(d_hat)
        sorted_weights = weights[:, sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights, axis=1)
        
    
        threshold = self.b / (self.b + self.h)
        mask = cumulative_weights >= threshold
        indices = np.argmax(mask, axis=1)
 
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
       

        return optimal_bandwidth
    
    def calculate_ko_cost(self, w, k_fold_sets):

        total_cost = 0.0

        for (X_train, d_train), (X_val, d_val) in k_fold_sets:

 
            q_values = self.kernel_optimization(X_train, d_train, X_val, w)


            cost = self.newsvendor_cost(q_values, d_val)
            total_cost += np.sum(cost)

        return total_cost
    
    def newsvendor_cost(self, q, d):

        tau = self.b / (self.h + self.b)
        return (1-tau) * np.maximum(q - d, 0) + tau * np.maximum(d - q, 0)

    def _find_optimal_s_l(self, k_fold_sets, s_values, l_values, optimal_w):

        min_cost = float('inf')
        optimal_s, optimal_l = s_values[0], l_values[0]
        

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


            q_values = [self.kernel_optimization(X_train, d_train, X_val[i, :], optimal_w) for i in range(len(d_val))]

            q_optimal = np.array(q_values)
            q_boosting = s*q_optimal + l

            cost = self.newsvendor_cost(q_boosting, d_val)
            total_cost += np.sum(cost)

        return total_cost

