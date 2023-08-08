import numpy as np
import pandas as pd
import random

def SquareWave(eps, x_grid):
    d = len(x_grid)
    x_step_size = x_grid[1]-x_grid[0]
    b = int(np.floor((eps*np.exp(eps)-np.exp(eps)+1)/
                    (2*np.exp(eps)*(np.exp(eps)-1-eps))*d))
    p = np.exp(eps)/((2*b+1)*np.exp(eps)+d-1)
    q = p/np.exp(eps)
    M = np.zeros((int(d+2*b),d))
    temp = np.concatenate((p*np.ones(2*b+1), q*np.ones(d-1)))
    for i in range(d):
        M[:,i] = np.roll(temp,i)
    a_grid = x_step_size*np.arange(-b,d+b)
    return a_grid, M

def GenRandResp(eps,x_grid):
    d = len(x_grid)
    p = np.exp(eps)/(np.exp(eps)+d-1)
    q = p/np.exp(eps)
    M = np.zeros((d,d))
    temp = np.concatenate((p*np.ones(1),q*np.ones(d-1)))
    for i in range(d):
        M[:,i] = np.roll(temp,i)
    return x_grid, M

def EM(M, a_count, eps):
    tau = 1e-3*np.exp(eps)
    _,d= M.shape
    a_count=np.reshape(a_count, (1,len(a_count)))
    p = np.ones((1,d))/d
    L_p = np.matmul(a_count,np.log(np.matmul(M,p.T)))
    while 1:
        temp = p*np.reshape(np.matmul(a_count,(M/np.matmul(M,p.T))),(1,d))
        p=temp/np.sum(temp)
        L = np.matmul(a_count,np.log(np.matmul(M,p.T)))
        if np.abs(L-L_p)< tau:
            break
        L_p=L
    return p

    
