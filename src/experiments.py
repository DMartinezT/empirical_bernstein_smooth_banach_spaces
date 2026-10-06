
import numpy as np

from utils import squared_norm_x_y

def psi_e(lam):
    return -lam - np.log(1 - lam)

def psi_g(lam):
    return lam**2 / 2 




class CI:
    def __init__(self, sample_size, sampler, B, kernel_type = "linear", alpha = 0.05, c1 = 0.5, c2 = 0.25):
        self.sample_size = sample_size
        self.sampler = sampler
        self.B = B
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        

    def __call__(self, random_seed):

        
        radii = np.zeros(self.sample_size)
        s2hat = np.zeros(self.sample_size+2)
        aux = np.sqrt(32*self.B**2 *np.log(2 / self.alpha))
        

        s2hat[0] = self.c2 * self.B**2
        
        X = self.sampler.sample_Y(n=1, random_seed=random_seed)
        normxy = squared_norm_x_y(X=X, kernel_type = self.kernel_type)  # "laplacian"
        s2hat[1] = s2hat[0] + normxy.get_sum_Kx()
        radii[0] = normxy.get_sum_Kx()
        
        for i in range(2, self.sample_size + 1):
            Xi = self.sampler.sample_Y(n=1)
            sqi = normxy(Xi)
            s2hat[i] = s2hat[i-1] + sqi
            radii[i-1] = sqi
            X = np.concatenate((X, Xi), 0)

        bernstein_radii = np.zeros(self.sample_size)
        bernstein_means = np.zeros(X.shape)
        hoeffding_radii = np.zeros(self.sample_size)
        hoeffding_means = np.zeros(X.shape)
        for i in range(self.sample_size):
            lambdas_b = np.zeros(i+1)
            for k in range(i+1):
                lambdas_b[k] = np.minimum(aux / np.sqrt(s2hat[k] * (i+1) / (k + 1)), self.c1) # 

            part1 = np.sum(radii[0:i+1]*psi_e(lambdas_b)) / (4*self.B)
            part2 = 4*self.B*np.log(2/self.alpha)
            bernstein_radii[i] = (part1 + part2)/np.sum(lambdas_b)
            bernstein_means[i, :] = (np.sum((lambdas_b*X[:i+1].T).T, 0).T / np.sum(lambdas_b)).T 

        res = np.concatenate((bernstein_means, np.expand_dims(bernstein_radii, axis = 1), hoeffding_means, np.expand_dims(hoeffding_radii, axis = 1)), 1)
        return res