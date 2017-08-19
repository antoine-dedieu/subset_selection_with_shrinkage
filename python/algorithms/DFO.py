import numpy as np
from   aux_DFO import *


# Implements the Discrete First Order (DFO) algorithm for l1, l2 or l2^2 regularizations



def DFO(type_penalization, X, y, K, llambda, beta_start=[], threshold_CV=1e-3, XTy=[], mu_max=0):
    
# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
# K, LLAMBDA       : sparsity and regularization parameters
# BETA_START       : warm start (or [])
# THRESHOLD_CV     : convergence threshold


#---Parameters
    N,P        = X.shape
    N_iter_max = 1e3                #maximum number of iterations

    if np.array(XTy).shape[0] == 0: XTy = np.dot(X.T, y)

    if mu_max == 0: mu_max = power_method(X)    #highest eigenvalue with power method


#---Intialization
    old_beta = -np.ones(P)
    beta     = beta_start if np.array(beta_start).shape[0]>0 else np.zeros(P)
    
    
#---We stop the main loop if the CV criterion is satisfied or after a maximum number of iterations
    test = 0
    while np.linalg.norm(beta-old_beta) > threshold_CV and test < N_iter_max: 
        test += 1
        old_beta = np.copy(beta)

    #---Gradient descent
        grad = beta - 1./mu_max * (np.dot(X.T, np.dot(X, beta)) - XTy)
        
    
    #---Set the P-k0 lowest coefficients to 0
        coefs_sorted = np.abs(grad).argsort()[::-1]
        
        for idx in coefs_sorted[K:]: beta[idx] = 0


    #---Apply the soft-thresholding operator for each of the K0 highest coefficients
        dict_thresholding = {'l1':   soft_thresholding_l1,
                             'l2':   soft_thresholding_l2,
                             'l2^2': soft_thresholding_l2_2}
        grad_thresholded = dict_thresholding[type_penalization](grad[coefs_sorted[:K]], llambda/mu_max) 
        beta[coefs_sorted[:K]]      = grad_thresholded


#---Solve restricted problem in the support
    beta, obj_val = solve_restricted_problem(type_penalization, X, y, llambda, beta)
    return beta, obj_val


