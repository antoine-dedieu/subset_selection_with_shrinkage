import sys
import os
import datetime
import random

from simulate_data import *

sys.path.append('../algorithms')
from neighborhood_continuation import *



# Runs an example 

## Simulation parameters
type_Sigma = 1

N, P = 50, 100
k0   = 7
rho  = 0.2
SNR  = 2

X, l2_X, real_beta, eps_train, eps_val, y_train, y_val = simulate_data(type_Sigma, N, P, k0, rho, SNR)


## Algorithm parameters
K_list       = range(15)
N_lambda     = 150 
threshold_CV = 1e-3

	
## First round
is_first_round = True
best_betas_l1, train_errors_list_l1, alpha_list_l1, betas_Lasso = neighborhood_continuation('l1',   X, y_train, K_list, N_lambda, is_first_round, threshold_CV)
best_betas_l2, train_errors_list_l2, alpha_list_l2, betas_Ridge = neighborhood_continuation('l2^2', X, y_train, K_list, N_lambda, is_first_round, threshold_CV)
