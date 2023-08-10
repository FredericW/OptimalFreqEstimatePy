import numpy as np
import scipy as sp
from scipy.optimize import linprog

def OptVariance(eps,x_grid,x_q): # the optimization problem
    a_grid = x_grid # we may change this in later update version
    da = a_grid.size
    dx = x_grid.size
    eEps = np.exp(eps)
    
    # the inequality constraints
    A_ub = np.zeros((da*(dx-1)*dx,da*dx)) # coefficient matrix
    counter=0
    for i in range(da):
        for j in range(dx):
            for k in range(dx):
                A = np.zeros((da,dx))
                if k!=j:
                    A[i,j] = 1
                    A[i,k]=-eEps
                    A = np.reshape(A,(1,dx*da))
                    A_ub[counter,:] = A
                    counter+=1
    b_ub = np.zeros((1,da*dx*(dx-1))) # limit vector

    # the equality constraints
    A_eq = np.zeros((dx+da,da*dx))
    for j in range(dx):
        temp = np.zeros((da,dx))
        temp[:,j] = np.ones(da)
        A_eq[j,:] = np.reshape(temp, (1,da*dx))
    for i in range(da):
        temp = np.zeros((da,dx))
        temp[i,:] = np.reshape(x_q,(1,dx))
        A_eq[dx+i,:] = np.reshape(temp,(1,da*dx))
    b_eq = np.concatenate((np.ones((1,dx)),x_q),axis = None) # limit vector

    # the obejetive funciton
    C = np.zeros((da,dx))
    for j in range(dx):
        C[:,j] = np.power(a_grid-x_grid[j],2)*x_q[0,j] # minimal cond. variance
    C = np.reshape(C,(1,da*dx))
    
    # limites
    bounds = list(zip(np.zeros(da*dx),np.ones(da*dx)))
    res = linprog(C, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq = b_eq, bounds=bounds)
    M = np.reshape(res.x,(da,dx))
    return a_grid, M

def Est2thenA3M()