import numpy as np
from gurobipy import *


# Solves the best subset regularized prolem to optimality 



def MIO_L0_LQ(type_penalization, X, y, K0, llambda, beta_start=[], time_limit=60):

#TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
#BETA_START       : possible warm start obtained with the prior neighborhood continuation

    MIO_L0_LQ = Model("MIO")
    MIO_L0_LQ.setParam('TimeLimit', time_limit)
    
    N,P = X.shape
    
#---Variables 
    beta     = np.array([MIO_L0_LQ.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
    z        = [MIO_L0_LQ.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]
    obj_val  = MIO_L0_LQ.addVar(name='obj_val')
    
    MIO_L0_LQ.update()    


#---Sparsity constraint
    MIO_L0_LQ.addConstr(quicksum(z) <= K0, "sparsity")


#---Big M constraint
    M=5*K0
    for i in range(P):
        MIO_L0_LQ.addConstr( beta[i] <= M*z[i], "max_beta_"+str(i))
        MIO_L0_LQ.addConstr(-beta[i] <= M*z[i], "min_beta_"+str(i))
        

#---Case l1
    if type_penalization == 'l1': 
        abs_beta = np.array([MIO_L0_LQ.addVar(lb=0, name="abs_beta_"+str(i)) for i in range(P)])
        MIO_L0_LQ.update()

        MIO_L0_LQ.addConstr(abs_beta[i] >=  beta[i], name='abs_beta_sup_'+str(i))
        MIO_L0_LQ.addConstr(abs_beta[i] >= -beta[i], name='abs_beta_inf_'+str(i)) 


#---Objective value
    dict_penalization = {'l1':   quicksum(abs_beta[i]     for i in range(P)), 
                         'l2^2': quicksum(beta[i]*beta[i] for i in range(P))}

    MIO_L0_LQ.setObjective(0.5*quicksum((y[i] - quicksum(X[i,k]*beta[k] for k in range(P)))*(y[i] - quicksum(X[i,k]*beta[k] for k in range(P))) for i in range(N)) + llambda*dict_penalization[type_penalization], GRB.MINIMIZE)

        
#---Warm start
    if np.array(beta_start).shape[0]>0:
        for i in range(P):
            beta[i].start = beta_start[i]
            z[i].start    = int(beta_start[i]!=0)
           
            if type_algo==1: abs_beta[i].start = abs(beta_start[i])
    
    
#---Solve
    MIO_L0_LQ.optimize()
    beta_MIO = [beta[i].x for i in range(P)]
    
    return np.round(beta_MIO,4), MIO_L0_LQ.ObjVal, MIO_L0_LQ










