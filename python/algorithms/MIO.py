import numpy as np
from gurobipy import *

from DFO import *


# Solves the best subset regularized prolem to optimality 



def MIO_L0_LQ(type_penalization, X, y, K, llambda, beta_start=[], time_limit=60, Big_M=0):

#TYPE PENALIZATION: type of regularization: 'l1', 'l2', 'l2^2'
    # -if 'l1' or 'l2^2', we write the problem as a MIO
    # -if 'l2', we consider its MISOCP formulation

#BETA_START       : possible warm start obtained with the prior neighborhood continuation
#BIG_M            : optional bound for infinite norm, usually returned with the DFO algorithm

    MIO_L0_LQ = Model("MIO")
    MIO_L0_LQ.setParam('TimeLimit', time_limit)
    
    N,P = X.shape
    
#---Variables 
    beta     = np.array([MIO_L0_LQ.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
    z        = [MIO_L0_LQ.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]
    obj_val  = MIO_L0_LQ.addVar(name='obj_val')
    
    MIO_L0_LQ.update()   

    # For later use
    aux_loss = quicksum((y[i] - quicksum(X[i,k]*beta[k] for k in range(P)))*(y[i] - quicksum(X[i,k]*beta[k] for k in range(P))) for i in range(N))


#---Big M constraint
    
    ##### CALL DFO #####
    if Big_M == 0:
        if len(beta_start) == 0: beta_start, _ = DFO(type_penalization, X, y, K, llambda)
        Big_M = 2*np.max(np.abs(beta_start))

    for i in range(P):
        MIO_L0_LQ.addConstr( beta[i] <= Big_M*z[i], "max_beta_"+str(i))
        MIO_L0_LQ.addConstr(-beta[i] <= Big_M*z[i], "min_beta_"+str(i))



#---Sparsity constraint
    MIO_L0_LQ.addConstr(quicksum(z) <= K, "sparsity")
     

#---Case l1
    if type_penalization == 'l1': 
        abs_beta = np.array([MIO_L0_LQ.addVar(lb=0, name="abs_beta_"+str(i)) for i in range(P)])
        MIO_L0_LQ.update()

        MIO_L0_LQ.addConstr(abs_beta[i] >=  beta[i], name='abs_beta_sup_'+str(i))
        MIO_L0_LQ.addConstr(abs_beta[i] >= -beta[i], name='abs_beta_inf_'+str(i)) 


#---Case l2 -> MISCOCP
    elif type_penalization == 'l2': 
        u = MIO_L0_LQ.addVar(lb=0, name="u")
        v = MIO_L0_LQ.addVar(lb=0, name="v")
        MIO_L0_LQ.update()

        MIO_L0_LQ.addConstr(aux_loss <=  u, name='u_constraint')
        MIO_L0_LQ.addConstr(quicksum(beta[i]*beta[i] for i in range(P)) <= v, name='v_constraint') 


#---Objective value
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
    
    
#---Solve
    MIO_L0_LQ.optimize()
    beta_MIO = [beta[i].x for i in range(P)]
    
    return np.round(beta_MIO,4), MIO_L0_LQ.ObjVal, MIO_L0_LQ










