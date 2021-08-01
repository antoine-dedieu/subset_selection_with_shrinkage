import numpy as np
from   sklearn.linear_model import Lasso
from   sklearn.linear_model import LinearRegression
import time

from DFO import *

## Supporting functions for the DFO algorithm. 

########## NEIGHBORHOOD CONTINUATION ALGORITHM ################
# Implements a neighborhood continuation method to make use of good warm-starts available from neighboring tuning parameters.
# Cycles through two sequences of parameters to produce a high-quality near-optimal regularization surface

def neighborhood_continuation(type_penalization, X, y, K_list, N_lambda, is_first_round=False, threshold_CV=1e-3, XTy=[], mu_max=0, betas_list=[], obj_vals_list=[], llambda_list=[], use_random_swap=False):

# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
# K_LIST           : list of sparsity constraints
# N_LAMBDA         : number of regularization parameters
# IS_FIRST_ROUND   : first round of the algorithm -> betas_list and obj_val_list are not defined
# BETAS_LIST       : list of estimators (potentialy already initialized)
# OBJ_VAL_LIST     : list of objective values (potentialy already initialized)
# LLAMBDA_LIST     : list of coefficients
# USE_RANDOM_SWAP  : boolean which indicates whether we use or not the random swaps
# THRESHOLD_CV     : convergence threshold for DFO algorithm


#---Parameters
    N,P   = X.shape
    start = time.time()
    norm_y = 0.5*np.linalg.norm(y)**2

    if np.array(XTy).shape[0] == 0: XTy = np.dot(X.T, y)
    
    if mu_max == 0: mu_max = power_method(X)    #highest eigenvalue with power method


#---Initialized if first round
    if is_first_round:

        # Results
        betas_list    = np.array([[np.zeros(P) for i in range(N_lambda)] for K in K_list])
        obj_vals_list = np.array([[norm_y      for i in range(N_lambda)] for K in K_list])
    
        # List of coefficients
        dict_llambda_max  = {'l1':np.max(np.abs(XTy)), 'l2':np.linalg.norm(XTy), 'l2^2':5*mu_max}
        llambda_max       = dict_llambda_max[type_penalization]
        llambda_list      = [llambda_max*(0.95)**i for i in range(N_lambda-1)]+[0]  # include 0 to compare with L0 without shrinkage



#---Initialize for K=P if first round
    if is_first_round:
        betas_Lasso_Ridge = Lasso_Ridge_path(type_penalization, X, y, llambda_list)
        K_max             = K_list[::-1][0]

        for loop in range(1, N_lambda):
            betas_list[K_max][loop], obj_vals_list[K_max][loop] = DFO(type_penalization, X, y, K_max, llambda_list[loop], betas_Lasso_Ridge[loop], threshold_CV, XTy, mu_max)
        
    else: betas_Lasso_Ridge = []


#---Cycle through K and lambda
    for K in K_list[1:]: 
        for loop in range(1, N_lambda): #no 0
            llambda = llambda_list[loop]

        #---Get all neighbors        
            beta_start_0, obj_val_0 = np.copy(betas_list[K][loop]), np.copy(obj_vals_list[K][loop])

            beta_start_1 = np.copy(betas_list[K-1][loop])
            beta_start_2 = np.copy(betas_list[K][loop-1])
            beta_start_3 = np.copy(betas_list[min(K+1, K_list[-1])][loop])
            beta_start_4 = np.copy(betas_list[K][min(loop+1, N_lambda-1)])


        #---Run DFO algorithm for all neighbors
            betas_nghbhd    = [beta_start_0]
            obj_vals_nghbhd = [obj_val_0]

            for beta_start in [beta_start_1, beta_start_2, beta_start_3, beta_start_4]:
                if use_random_swap: beta_start = shuffle_half_support(beta_start, K, P)

                beta, obj_val = DFO(type_penalization, X, y, K, llambda, beta_start, threshold_CV, XTy, mu_max)
                betas_nghbhd.append(beta)
                obj_vals_nghbhd.append(obj_val)
            

        #---Keep estimator with lowest objective value
            argmin                 = np.argmin(obj_vals_nghbhd)
            betas_list[K][loop]    = np.copy(betas_nghbhd[argmin])
            obj_vals_list[K][loop] = np.copy(obj_vals_nghbhd[argmin])

                                                     
    print 'Time for neighborhood continuation algorithm with '+type_penalization+' regularization: '+str(round(time.time()-start,2))
    return betas_list, obj_vals_list, llambda_list, betas_Lasso_Ridge





############## LASSO OR RIDGE REGULARIZATION PATH ################


# Compute the regularization path for K=P using scikit_learn for Lasso and an efficient SVD or Ridge

def Lasso_Ridge_path(type_penalization, X, y, llambda_list):

#TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
#LLAMBDA_LIST     : sequence of regularization coefficients


#---Result
    betas_Lasso_Ridge  = [] 
    N,P  = X.shape
    beta = np.zeros(P)

#---Compute SVD once for Lasso with alpha=0 and Ridge 
    U, d, V     = np.linalg.svd(X, full_matrices=False)
    UTy         = np.dot(U.T, y)
    LS_solution = np.dot(V.T, np.dot(np.diag(1./d), UTy))


#---Coefficients in decreasing order
    for llambda in llambda_list:
        
        if llambda > 0:
            if type_penalization == 'l1':
                ls  = Lasso(alpha=llambda/float(N), max_iter=100000)
                ls.fit(X, y)
                beta  = ls.coef_
            
            elif type_penalization == 'l2':
                beta, _ = DFO('l2', X, y, P, llambda, beta_start=beta)

            elif type_penalization == 'l2^2':
                diag    = d/(d**2 + 2*llambda)
                beta    = np.dot(V.T, np.dot(np.diag(diag), UTy)) #Ridge solution with SVD: http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf

        else:
            beta  = LS_solution 

        betas_Lasso_Ridge.append(np.copy(beta))  #useful for later comparison

    return betas_Lasso_Ridge




############## RANDOM SWAPS ################


# Swap half of the support of the actual estimator to encourage exploration

def shuffle_half_support(current_beta, K0, P):
    
    beta    = np.copy(current_beta)
    support = np.where(beta!=0)[0]
    np.random.shuffle(support)

    beta[np.random.randint(P,size=int((K0+1)/2))] = beta[support[:int((K0+1)/2)]]
    beta[support[:int((K0+1)/2)]] = np.zeros(int((K0+1)/2))

    return beta

