# Subset Selection with Shrinkage: Sparse Linear Modeling when the SNR is low 

<!---
 ## Getting Started
 ## Algorithms
-->

### Antoine Dedieu, Rahul Mazumder, and Peter Radchenko

## Introduction

We consider a regularized version of the canonical best subset estimator, which is given by the following optimization problem

```
minimize 0.5*\| y - X \beta \|_2^2 + \lambda \|\beta\|_{q}^q
subject to \| \beta \|_0  <= K,
```

where \lambda and K are two regularization parameters, and q is either 1 or 2. We call this the L0-LQ estimator where 
L0 corresponds to the cardinality constraint and LQ the penalty function that is meant to impart shrinkage. Above, we estimate 
the regression coefficients (\beta); where \lambda, K and q are provided by the user. 

The optimization problem above can be expressed as a Mixed Integer Optimization (MIO) formulation. To obtain good upper bounds, we use low complexity Discrete First Order (DFO) methods. A neighborhood continuation heuristic using warm-starts across the tuning parameters (\lambda, K) is also proposed. 
The solutions obtained from the above methods serve as warm-starts for our MIO-based framework. For additional details on the algorithm and statistical properties of the L0-LQ estimator, please see our manuscript *Subset selection with shrinkage: Sparse linear modeling when the SNR is low* ([link](https://arxiv.org/abs/1708.03288))


## Implementation
The Discrete First Order (DFO) algorithm written in Python can be used as a standalone algorithm. 
The MIO formulation is solved with Gurobi's mixed integer programming solver. 

## Demo Example
To see a demo example on how to use our algorithm, please refer to example.py file located at 

python/example/example.py ([link](https://github.com/antoine-dedieu/subset_selection_with_shrinkage/blob/master/python/example/example.py))


<!---
Our toolkit is implemented in Python.

We propose the two following implementations:

1. [Python]

We implement the Discrete First Order (DFO) algorithm and the Neighborhood Continuation heuristic respectively presented in Sections 2.2 and 2.3 of the paper. We also propose an Gurobi MIO solver for the regularized best subset problem as defined in Section 2.1. The settings of the data generation and experiments are as described in Section 5.

2. [Julia]


### Citation

Please use the following citation to cite this work

```

-->
