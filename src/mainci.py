import numpy as np
from tqdm import tqdm

from experiments import CI
from distributions import uniform, beta, bernoulli, bernoulli_mixture, normal



if __name__ == "__main__":

    n_experiments = 500
    sample_size = 4000

    experiment_type_list = ["beta", "bernoulli"]
    d_list = [5] # d = 1 for RKHS experiments
    kernel_type = "linear" # "rbf" for RKHS experiments

    for d in d_list:
        for experiment_type in experiment_type_list:
            print(f"Experiment type: {experiment_type} with d={d}.")
        
            if experiment_type == "uniform":
                sampler = uniform(d=d)
                B = np.sqrt(d)    
                experiment = CI(sample_size=sample_size, sampler=sampler, B=B, kernel_type = kernel_type, alpha = 0.05, c1 = 0.5, c2 = .25)
            
            
            elif experiment_type == "bernoulli":
                sampler = bernoulli(d=d)
                B = np.sqrt(d)    
                experiment = CI(sample_size=sample_size, sampler=sampler, B=B, kernel_type = kernel_type, alpha = 0.05, c1 = 0.5, c2 = 0.25)


                
            T_reshaped = np.zeros((n_experiments, sample_size, (d+1)*2))
            for random_seed in tqdm(range(n_experiments)):
                T_reshaped[random_seed, :] =  experiment(random_seed)
            
            with open("../data/bernstein_hoeffding/CI/" + experiment_type + str(d) + ".npy", 'wb') as f:
                np.save(f, T_reshaped)