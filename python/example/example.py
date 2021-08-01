import sys
import os
import datetime
import random

from simulate_data import *

sys.path.append('../algorithms')
from neighborhood_continuation import *
from MIO import *


# Runs an example to illustrate using our code.

## Generate synthetic data. Simulation parameters

# Type of covariance matrix of covariates [see paper for details]
type_Sigma = 1

N, P = 50, 100
k0   = 7
rho  = 0.2
SNR  = 2

X, l2_X, real_beta, eps_train, eps_val, y_train, y_val = simulate_data(type_Sigma, N, P, k0, rho, SNR)


## Algorithm parameters
## K_list -> cardinality constraints. N_lambda -> # continuous penalty parameters
K_list       = range(15)
N_lambda     = 150 
threshold_CV = 1e-3


## DFO: Uses heuristics to compute good solutions to the regularized subsets problem on a 2D grid of tuning parameters.	
## First round
is_first_round = True
best_betas_l1,   train_errors_list_l1,   alpha_list_l1,   betas_Lasso = neighborhood_continuation('l1',   X, y_train, K_list, N_lambda, is_first_round, threshold_CV)
best_betas_l2,   train_errors_list_l2,   alpha_list_l2,   _           = neighborhood_continuation('l2',   X, y_train, K_list, N_lambda, is_first_round, threshold_CV)
##best_betas_l2_2, train_errors_list_l2_2, alpha_list_l2_2, betas_Ridge = neighborhood_continuation('l2^2', X, y_train, K_list, N_lambda, is_first_round, threshold_CV)



## MIO: Run the MIO solver to obtain solutions to the L0-LQ Problem
MIO_L0_LQ('l1',   X, y_train, 7, 0.1, time_limit=30)
MIO_L0_LQ('l2',   X, y_train, 7, 0.1, time_limit=30)
##MIO_L0_LQ('l2^2', X, y_train, 7, 0.1, time_limit=30)

