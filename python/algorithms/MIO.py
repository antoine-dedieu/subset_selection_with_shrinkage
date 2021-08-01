import numpy as np
from gurobipy import *

from DFO import *

# Solves the best subset regularized prolem to optimality.
# This script formulates our L0-LQ estimator as a mixed integer optimization problem, which can be solved to optimality via branch-and-bound/cut. We use Gurobi's MIO solver. 
# Please make sure that you can call Gurobi's solver from your machine.
# In addition to the L0-LQ estimator, we also offer the option of computing a regularized best-subset problem where the regularizer is the ridge penalty (i.e., l2^2).


def MIO_L0_LQ(type_penalization, X, y, K, llambda, beta_start=[], time_limit=60, Big_M=0):

# Function arguments:    
#type_penalization: which type of regularization to use in the penalty/objective: 'l1', 'l2', 'l2^2'
    # -if 'l1' or 'l2^2', we write the problem as a mixed integer quadratic optimization problem
    # -if 'l2', we consider a mixed integer second order conic programming formulation. 

#time_limit       : how long should the MIP solver run? We took time_limit=60 as a placeholder only. We recommend using ~30 mins for p~1000.    
#beta_start       : [optional] possible warm start obtained with our discrete first order (DFO) methods. Supplying this can speed up the overall computation time. 
#Big_M            : [optional] bound on the magnitudes of the regression coefficients. If a reliable bound is not available to the user, we recommend using "Big_M=0", 
#                   so that the algorithm can compute a BigM value based on running a DFO method prior to running the MIO-algorithm.  


# Setup the MIO model in Gurobi/Python

MIO_L0_LQ = Model("MIO")
    MIO_L0_LQ.setParam('TimeLimit', time_limit)
    
    N,P = X.shape
    
#---Define the decision variables 
    beta     = np.array([MIO_L0_LQ.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
    z        = [MIO_L0_LQ.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]
    obj_val  = MIO_L0_LQ.addVar(name='obj_val')
    
    MIO_L0_LQ.update()   

    # Setting up the squared residual component of the objective, to be used later to define our objective function
    aux_loss = quicksum((y[i] - quicksum(X[i,k]*beta[k] for k in range(P)))*(y[i] - quicksum(X[i,k]*beta[k] for k in range(P))) for i in range(N))


#---Big M constraint
    
    ##### CALL DFO to estimate a BigM value #####
    if Big_M == 0:
        if len(beta_start) == 0: beta_start, _ = DFO(type_penalization, X, y, K, llambda)
        Big_M = 2*np.max(np.abs(beta_start))

    for i in range(P):
        MIO_L0_LQ.addConstr( beta[i] <= Big_M*z[i], "max_beta_"+str(i))
        MIO_L0_LQ.addConstr(-beta[i] <= Big_M*z[i], "min_beta_"+str(i))



#---Sparsity constraint
    MIO_L0_LQ.addConstr(quicksum(z) <= K, "sparsity")
     

#---Case l1 [type of penalty function in the objective]
    if type_penalization == 'l1': 
        abs_beta = np.array([MIO_L0_LQ.addVar(lb=0, name="abs_beta_"+str(i)) for i in range(P)])
        MIO_L0_LQ.update()

        MIO_L0_LQ.addConstr(abs_beta[i] >=  beta[i], name='abs_beta_sup_'+str(i))
        MIO_L0_LQ.addConstr(abs_beta[i] >= -beta[i], name='abs_beta_inf_'+str(i)) 


#---Case l2 [type of penalty function in the objective] -> this leads to a SOCP constraint
    elif type_penalization == 'l2': 
        u = MIO_L0_LQ.addVar(lb=0, name="u")
        v = MIO_L0_LQ.addVar(lb=0, name="v")
        MIO_L0_LQ.update()

        MIO_L0_LQ.addConstr(aux_loss <=  u, name='u_constraint')
        MIO_L0_LQ.addConstr(quicksum(beta[i]*beta[i] for i in range(P)) <= v, name='v_constraint') 


#---Setting up the Objective value of the problem
    loss = u if type_penalization == 'l2' else aux_loss
    
    if type_penalization == 'l1':
        pen = quicksum(abs_beta[i] for i in range(P))
    elif type_penalization == 'l2':
        pen = v
    elif type_penalization == 'l2^2':
        pen = quicksum(beta[i]*beta[i] for i in range(P))

    MIO_L0_LQ.setObjective(0.5*loss + llambda*pen, GRB.MINIMIZE)


        
#---Warm start
    if np.array(beta_start).shape[0]>0:
        for i in range(P):
            beta[i].start = beta_start[i]
            z[i].start    = int(beta_start[i]!=0)
           
            if type_penalization==1: abs_beta[i].start = abs(beta_start[i])
    
    
#---Solve the Model that we set up
    MIO_L0_LQ.optimize()
    beta_MIO = [beta[i].x for i in range(P)]
    
    return np.round(beta_MIO,4), MIO_L0_LQ.ObjVal, MIO_L0_LQ










