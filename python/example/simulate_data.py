import numpy as np
import random


# Generates the data: we first simulate a N*P design matrix with each row being a multivariate normal with mean real_beta and covariance Sigma
    # -if type_Sigma == 1, real_beta = 1 for k0 equi-spaced coefficients and Sigma_i,j = rho^|i-j|
    # -if type_Sigma == 2, real_beta = 1 for the first k0 coefficients   and Sigma_i,j = rho

# We then create two centered independent noise vectors eps_train and eps_val with standard deviation defined using the SNR of the problem
# The train and validation vectors satisfy y_train = X_train*real_beta + eps_train and y_train = X_train*real_beta + eps_val


def simulate_data(type_Sigma, N, P, k0, rho, SNR, seed_X=-1, seed_eps=-1):

#INPUTS:
# -TYPE_SIGMA: Setting for correlation matrix
# -N,P:        Size of the design matrix
# -RHO:        Correlation coefficient
# -SNR:        Signal to Noise ratio
# -SEED_X, SEED_EPS: Two seeds needed 



#OUTPUTS:
# -X          : design matrix of size N*P with l2 unit norm columns
# -l2_X       : list of l2 norms of the columns
# -REAL_BETA  : vector to estimate 
# -Y_TRAIN    : training vector of size N,   with noise eps_train
# -Y_VAL      : validation vector of size N, with noise eps_val



    if seed_X == -1:   seed_X  = random.randint(0,1000)
    if seed_eps == -1: seed_eps= random.randint(0,1000)    


#-------------------------Beta---------------------------
    
    real_beta = np.zeros(P)

    if   type_Sigma == 1: idx = [(2*i+1)*P/(2*k0) for i in range(k0)] #k0 equi-spaced coefficients
    elif type_Sigma == 2: idx = range(k0)

    real_beta[idx] = np.ones(k0)


#-------------------------X-----------------------------
    
    np.random.seed(seed=seed_X)

    # For type_Sigma==1, we use a Cholesky decomposition
    if type_Sigma == 1:
        Sigma = [[rho**(abs(i-j)) for i in range(P)] for j in range(P)]

        L = np.linalg.cholesky(Sigma)
        u = np.random.normal(size=(P,N))
        X = np.dot(L, u).T

    # For type_Sigma==2, we use a fast method
    elif type_Sigma == 2:
        X0 = np.random.normal(size=(N,1))
        Xi = np.random.normal(size=(N,P))
        X  = np.zeros((N,P))

        for i in range(N): X[i,:] = math.sqrt(rho)*X0[i] + math.sqrt(1-rho)*Xi[i,:]
        


#-------------------------Eps, y---------------------------

    np.random.seed(seed=seed_eps)
    X_real_beta = np.dot(X, real_beta)

    std_eps   = np.sqrt(X_real_beta.var() / float(SNR**2)) #SNR is the ratio between std(X.beta) and std(epsilon)
    eps_train = np.random.normal(0, std_eps, N)
    y_train   = X_real_beta + eps_train

    eps_val = np.random.normal(0, std_eps, N)
    y_val   = X_real_beta + eps_val



#---Normalize all the X columns
    l2_X   = []
    for i in range(P):
        l2      = np.linalg.norm(X[:,i])
        X[:,i] /= l2
        l2_X.append(l2)  


    print 'Data generated for N='+str(N)+' P='+str(P)+' k0='+str(k0)+' SNR='+str(SNR)+' Rho='+str(rho)+' Sigma='+str(type_Sigma)+' seed_X='+str(seed_X)+' seed_eps='+str(seed_eps)
    return X, l2_X, real_beta, eps_train, eps_val, y_train, y_val







