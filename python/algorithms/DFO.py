import numpy as np
from aux_DFO import *

# Implements the Discrete First Order (DFO) algorithm for l1, l2 penalty functions with a cardinality constraint 
# min  0.5*\| y - X \beta \|_2^2 + \lambda \|\beta\|_{q}  subject to \| \beta \|_0  <= K
# We also have an additional option for a ridge penalty (aka l2^2) instead of an l2 penalty in the objective:
# min  0.5*\| y - X \beta \|_2^2 + \lambda \|\beta\|^2_{2}  subject to \| \beta \|_0  <= K
# The penalty in the objective is denoted by "type_penalization" [function argument]

def DFO(type_penalization, X, y, K, llambda, beta_start=[], threshold_CV=1e-3, XTy=[], mu_max=0, solve_support=True):
    
# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
# K, LLAMBDA       : sparsity and regularization parameters
# BETA_START       : warm start if available (or [] i.e., no warm-start is provided)
# THRESHOLD_CV     : convergence threshold i.e., we stop if "l2-norm(beta-old_beta) <= THRESHOLD_CV". Note we stop if the number of iterations is > N_iter_max=1e3
# SOLVE_SUPPORT    : false only when solving on support for l2

# Other arguments [we recommend to keep the default setting]: 
# mu_max=0 : this computes the maximum eigenvalue of X'X via the power method. This is required to choose
#                                                       a suitable stepsize for the algorithm
# solve_support=True : does a polishing operation on the support obtained upon convergence. That is, we solve the convex problem restricted to the support.

#---Parameters
    N,P        = X.shape
    N_iter_max = 1e3                #maximum number of iterations
    obj_val    = 0

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
        
    
    #---Set the P-k0 lowest coefficients to 0 [hard thresholding step]
        coefs_sorted = np.abs(grad).argsort()[::-1]
        
        for idx in coefs_sorted[K:]: beta[idx] = 0


    #---Apply the thresholding operation [depending upon the penalty function] for each of the K0 coefficients that survive the hard thresholding operation above
        dict_thresholding = {'l1':   soft_thresholding_l1,
                             'l2':   soft_thresholding_l2,
                             'l2^2': soft_thresholding_l2_2}
        grad_thresholded = dict_thresholding[type_penalization](grad[coefs_sorted[:K]], llambda/mu_max) 
        beta[coefs_sorted[:K]]      = grad_thresholded


#---Solve restricted problem in the support
    if solve_support: beta, obj_val = solve_restricted_problem(type_penalization, X, y, llambda, beta)
    
    return beta, obj_val


