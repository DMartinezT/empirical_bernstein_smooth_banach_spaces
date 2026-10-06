import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class squared_norm_x_y():

    def __init__(self, X, kernel_type = "linear"):

        self.kernel_type = kernel_type 
        self.X = X
        self.sum_Kx = np.sum(pairwise_kernels(self.X, metric=self.kernel_type))
        self.n = 1
        
    def __call__(self, Xi):

        Kxxi = pairwise_kernels(self.X, Xi, metric=self.kernel_type)
        Kxi = pairwise_kernels(Xi, metric=self.kernel_type)
        

        res = self.sum_Kx/(self.n*self.n) + np.sum(Kxi) - 2*np.sum(Kxxi)/self.n
        
        self.n += 1
        self.sum_Kx += np.sum(Kxi) + 2*np.sum(Kxxi)
        self.X = np.concatenate((self.X, Xi), 0)

        return res

    def get_sum_Kx(self):
        return self.sum_Kx

        