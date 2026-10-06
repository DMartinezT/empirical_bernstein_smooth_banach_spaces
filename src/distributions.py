import numpy as np

class uniform():
    def __init__(self, d = 2):
        self.d = d

    def sample_Y(self, n, random_seed = None):
        if random_seed is not None: 
            np.random.seed(random_seed)
        Y = np.random.uniform(low = -1, high = 1, size = (n, self.d))
        return Y 

class bernoulli():
    def __init__(self, d = 2, p = 0.5):
        self.d = d
        self.p = p

    def sample_Y(self, n, random_seed = None):
        if random_seed is not None: 
            np.random.seed(random_seed)
        Y = 2*np.random.binomial(n = 1, p = self.p, size = (n, self.d))-1
        return Y 
