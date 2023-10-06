import numpy as np
import utilities
from a3mV2 import opt_variance
import math
np.set_printoptions(precision=3)

def generate_synthetic_data(data_type="GAUSSIAN", 
                            n=100000, low=0, high=1, beta=1):
    # generate data
    if data_type == "GAUSSIAN":
        data = np.random.normal(0, 1.0, n)
    elif data_type == "EXPONENTIAL":
        data = np.random.exponential(1, n)
        low = 0
    elif data_type == "GAMMA":
        data = np.random.gamma(2, 2, n)
        low = 0
    elif data_type == "BETA":
        data = np.random.beta(0.5, 0.5, n)
        low = 0
        high = 1
    elif data_type == "UNIFORM":
        data = np.random.uniform(0, 1, n)
    elif data_type == "GAUSSMIX":
        data_1 = np.random.normal(-3, 1.0, n)
        data_2 = np.random.normal(3, 1.0, n)
        p = 0.5 * np.ones(n) 
        ber = np.random.binomial(1, p, size=None)
        data = ber*data_1 + (1-ber)*data_2
    else:
        raise NotImplementedError()
    # clip data to [low,high]
    clipped_data = np.clip(data, low, high)
    # linearly map data from [low,high] to [0,beta]
    """ Let linear map be: y = ax + b
        Then: 
        -beta = a*low + b and beta = a*high+b
        We have:
    """
    a = beta / (high-low)
    b = beta - beta*high / (high-low)
    mapped_data = a*clipped_data + b
    return mapped_data



def M_to_pool(M, a_grid, N_pool):
    _,dx =M.shape
    perturbed_pool = np.zeros((N_pool,dx))
    for j in range(dx):
        p=M[:,j]/np.sum(M[:,j])
        uni_noise = np.random.uniform(low=-1/2/dx,high = 1/2/dx,size=N_pool)
        grid_noise = np.random.choice(a=a_grid,p=p,size=N_pool)+np.random.uniform(low=-1/2/dx,high = 1/2/dx,size=N_pool)
        perturbed_pool[:,j] = uni_noise + grid_noise
    return perturbed_pool




def DP_dist_estimation(data, bin_max,d_est, est_type, eps, repeat, q_est):
    # set the bins for histogram
    beta=1
    bin_idxs_est = np.linspace(beta/d_est/2, beta-beta/d_est/2, d_est)

    # obtain the histogram and frequency
    histo_true_est,_ = np.histogram(a=data, range=(0,bin_max), bins=d_est)
    q_true = histo_true_est/np.sum(histo_true_est)

    # generate the transition prbability matrix
    N_pool = 100000
    if est_type == 'sw':
        a_grid, M = utilities.square_wave(eps,bin_idxs_est)
    elif est_type == 'grr':
        a_grid, M = utilities.general_rr(eps,bin_idxs_est)
    elif est_type=='aaa':
        a_grid=bin_idxs_est
        _, sol = opt_variance(eps, a_grid, q_est)
        if sol.success == True:
            M = np.reshape(sol.x,(d_est,d_est))
    else:
        raise NotImplementedError('invalid est_type!')
    
    # generate random pool from the matrix
    perturbed_pool = M_to_pool(
        M=M, a_grid=a_grid, N_pool=N_pool)
    
    mse_est = utilities.M_to_var(M,a_grid,bin_idxs_est,q_true)

    wass_est = 0
    q_est = np.zeros((1,d_est))
    for _ in np.arange(repeat):
        data_perturbed=[]
        for i in np.arange(len(histo_true_est)):
            temp = np.random.choice(a = perturbed_pool[:,i], size = histo_true_est[i])
            data_perturbed = np.concatenate((data_perturbed,temp))
        _,hist_perturbed = np.unique(data_perturbed,return_counts=True)
        if est_type!="aaa":
            temp = utilities.EM(M,hist_perturbed,eps)
        else:
            temp = hist_perturbed/data_perturbed.size
        q_est +=temp
        wass_est += utilities.wass_dist(temp,q_true)
    q_est = q_est/repeat
    wass_est = wass_est/repeat

    return mse_est, wass_est, q_est