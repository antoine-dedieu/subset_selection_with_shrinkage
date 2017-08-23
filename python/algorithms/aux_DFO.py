import numpy as np
from   sklearn.linear_model import LinearRegression
from   sklearn.linear_model import Lasso 
from   sklearn.linear_model import Ridge 

import DFO as DFO


# Support functions for the Discrete First Order (DFO) algorithm


############## HIGHEST EIGENVALUE OF XTX ################

# Classical power method for fast computation of the highest eigenvalue of XTX with p large
# Beta is randomly initialized. Then at every iteration, it is multiplied by XTX and normalized

def power_method(X):
    P = X.shape[1]

#---Compute the highest eigenvector
    highest_eigvctr     = np.random.rand(P) #random intialization
    old_highest_eigvctr = 1e6*np.ones(P)
    
    while np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2:   #CV criterion
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))   #p is large
        highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
    
#---Deduce the highest eigenvalue
    X_highest_eigval = np.dot(X, highest_eigvctr)
    highest_eigval   = np.dot(X_highest_eigval.T, X_highest_eigval)/np.linalg.norm(highest_eigvctr)
    
    return highest_eigval





############## SOFT THRESHOLDING OPERATORS ################

# Solve argmin 0.5 \| \beta - u \|_2^2 + llambda L(\beta) with L \in {l1, l2, l2^2}

def soft_thresholding_l1(u, llambda):
    return np.array([np.sign(u_i)*max(0, abs(u_i)-llambda) for u_i in u])


def soft_thresholding_l2(u, llambda):
    l2_norm = np.linalg.norm(u)
    return max(0, l2_norm-llambda)/(l2_norm+1e-10)*u

    
def soft_thresholding_l2_2(u, llambda):
    return np.array([u_i/(1.+2*llambda) for u_i in u])






############## SOLVE RETRICTED PROBLEM ON SUPPORT ################

def solve_restricted_problem(type_penalization, X, y, llambda, beta):

# TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
    # - for 'l1' or 'l2^2', we use Lasso or Ridge solved with scikit learn
    # - for 'l2', we use a soft thresholding gradient descent on the support
#BETA              : current estimator with support of size K
    
    N,P     = X.shape
    support = np.where(beta!=0)[0]


#---Support not empty
    if support.shape[0] > 0:

    #---Data resticted on support
        X_support = np.array([X[:,support[i]] for i in range(len(support))]).T
        

    #---Solve restricted problem

        if type_penalization == 'l2':
        ### FOR L2 CALL DFO WITHOUT SOLVING ON SUPPORT
            beta_support, _ = DFO.DFO('l2', X_support, y, len(support), llambda, beta_start=beta[support], solve_support=False)

        else:
            if llambda == 0:
                estimator = LinearRegression()
            
            elif type_penalization in {'l1', 'l2^2'}:
            ### CAREFULL : coefficients
                dict_estimator = {'l1':  Lasso(alpha=llambda/float(N)),#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
                                  'l2^2': Ridge(alpha=2.*llambda, fit_intercept=False, solver='svd')} #http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
                estimator = dict_estimator[type_penalization]

            estimator.fit(X_support, y)
            beta_support  = np.array(estimator.coef_)
            


    #---Compute loss
        beta[support]     = beta_support
        obj_val           = 0.5*np.linalg.norm(y - np.dot(X_support, beta_support) )**2
        dict_penalization = {'l1': np.sum(np.abs(beta_support)), 'l2': np.linalg.norm(beta_support), 'l2^2': np.linalg.norm(beta_support)**2}
        obj_val          += llambda*dict_penalization[type_penalization]


#---Support empty
    else:
        obj_val = 0.5*np.linalg.norm(y)**2

    return beta, obj_val





