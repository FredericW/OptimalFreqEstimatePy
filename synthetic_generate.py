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



def generate_perturbed_pool(M, a_grid, N_pool):
    _,dx =M.shape
    perturbed_pool = np.zeros((N_pool,dx))
    for j in range(dx):
        p=M[:,j]/np.sum(M[:,j])
        perturbed_pool[:,j] = np.random.choice(a=a_grid,p=p,size=N_pool)
    return perturbed_pool



def estimate_distribution(eps,est_type,M,histo_true,pool,repeat):
    q_est = np.zeros((1,histo_true.size))
    for k in np.arange(repeat):
        data_perturbed=[]
        for i in np.arange(len(histo_true)):
            temp = np.random.choice(a = pool[:,i], size = histo_true[i])
            data_perturbed = np.concatenate((data_perturbed,temp))
        _,hist_perturbed = np.unique(data_perturbed,return_counts=True)
        if est_type!="aaa":
            temp = utilities.EM(M,hist_perturbed,eps)
        else:
            temp = hist_perturbed/data_perturbed.size
        q_est +=temp
    q_est = q_est/repeat
    return q_est



def DP_dist_estimation(data, bins, bin_idxs, range, 
                       est_type, eps, est_repeat, test_repeat):
    print('eps=',eps)
    histo_true,_ = np.histogram(a=data, range=range, bins=bins)
    N_pool = bins*10000
    if est_type == 'sw':
        a_grid, M_est = utilities.square_wave(eps,bin_idxs)
    elif est_type == 'grr':
        a_grid, M_est = utilities.general_rr(eps,bin_idxs)
    else:
        raise NotImplementedError()
    perturbed_pool_est = generate_perturbed_pool(
        M=M_est, a_grid=a_grid, N_pool=N_pool)

    q_est_initial = estimate_distribution(eps=eps,est_type=est_type,
                                M=M_est,
                                histo_true=histo_true, 
                                pool=perturbed_pool_est,
                                repeat=est_repeat)

    _, M_aaa = opt_variance(eps, bin_idxs, q_est_initial)

    perturbed_pool_aaa = generate_perturbed_pool(
        M=M_aaa, a_grid=bin_idxs, N_pool=N_pool)
    
    q_true = histo_true/sum(histo_true)

    var_aaa = utilities.M_to_var(M_aaa,bin_idxs,bin_idxs,q_true)
    
    var_est = utilities.M_to_var(M_est,a_grid,bin_idxs,q_true)

    q_aaa = estimate_distribution(eps=eps,est_type="aaa",
                                M=M_aaa,
                                histo_true=histo_true, 
                                pool=perturbed_pool_aaa,
                                repeat=test_repeat)
    wass_aaa = utilities.wass_dist(q_aaa,q_true)


    q_est = estimate_distribution(eps=eps,est_type=est_type,
                                M=M_est,
                                histo_true=histo_true, 
                                pool=perturbed_pool_est,
                                repeat=test_repeat)  
    wass_est= utilities.wass_dist(q_est,q_true)
    return var_aaa, var_est, wass_aaa, wass_est

def DP_dist_estimation_batch(data, bins, bin_idxs, range, 
                       est_type, eps, est_repeat, test_repeat, portion=0.1):
    print('eps=',eps)
    histo_true,_ = np.histogram(a=data, range=range, bins=bins)
    N_pool = bins*10000
    if est_type == 'sw':
        a_grid, M_est = utilities.square_wave(eps,bin_idxs)
    elif est_type == 'grr':
        a_grid, M_est = utilities.general_rr(eps,bin_idxs)
    else:
        raise NotImplementedError()
    perturbed_pool_est = generate_perturbed_pool(
        M=M_est, a_grid=a_grid, N_pool=N_pool)
    
    data_batch = np.random.choice(a=data, replace=False, 
                                  size = math.ceil(data.size*portion))
    histo_batch,_ = np.histogram(a=data_batch, range=range, bins=bins)
    q_est_initial = estimate_distribution(eps=eps,est_type=est_type,
                                M=M_est,
                                histo_true=histo_batch, 
                                pool=perturbed_pool_est,
                                repeat=est_repeat)

    _, M_aaa = opt_variance(eps, bin_idxs, q_est_initial)

    perturbed_pool_aaa = generate_perturbed_pool(
        M=M_aaa, a_grid=bin_idxs, N_pool=N_pool)
    
    q_true = histo_true/sum(histo_true)

    var_aaa = utilities.M_to_var(M_aaa,bin_idxs,bin_idxs,q_true)
    
    var_est = utilities.M_to_var(M_est,a_grid,bin_idxs,q_true)

    q_aaa = estimate_distribution(eps=eps,est_type="aaa",
                                M=M_aaa,
                                histo_true=histo_true, 
                                pool=perturbed_pool_aaa,
                                repeat=test_repeat)
    wass_aaa = utilities.wass_dist(q_aaa,q_true)


    q_est = estimate_distribution(eps=eps,est_type=est_type,
                                M=M_est,
                                histo_true=histo_true, 
                                pool=perturbed_pool_est,
                                repeat=test_repeat)  
    wass_est= utilities.wass_dist(q_est,q_true)
    return var_aaa, var_est, wass_aaa, wass_est