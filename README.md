# Subset Selection with Shrinkage 

## Getting Started

## Algorithms

### Our framework

We consider the regularized best subset estimator defined as 
```
minimize 0.5*\| y - X \beta \|_2^2 + \lambda l_q (\beta)
subject to \| \beta \|_0  <= K,
```
where \lambda and K are two regularization coefficients, and l_q is the l_1, l_2 or l_2^2 regularization. This problem can be expressed with a Mixed Integer Optimization (MIO) formulation. To obtain good upper bounds,  we solve the problem to near-optimality for a fixed couple of parameters using a low complexity Discrete First Order (DFO) algorithm. As this method is sensitive to the initialization, we propose a neighborhood continuation heuristic which cycles through two sequences of parameters to build a regularization surface of near-optimal high-quality solutions.



### Implementations

We propose the two following implementations:

1. [Python]

We implement the Discrete First Order (DFO) algorithm and the Neighborhood Continuation heuristic respectively presented in Sections 2.2 and 2.3 of the paper. We also propose an Gurobi MIO solver for the regularized best subset problem as defined in Section 2.1. The settings of the data generation and experiments are as described in Section 5.

2. [Julia]


### Citation

Please use the following citation to cite this work

```
