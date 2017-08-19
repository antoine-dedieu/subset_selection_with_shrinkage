#Subset Selection with Shrinkage

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example



## Our approach

### Background

This page contains several sample implementations of an estimation procedure for FA as described in [Bertsimas, Copenhaver, and Mazumder, "Certifiably Optimal Low Rank Factor Analysis", Journal of Machine Learning Research 18 (2017) ("BCM17")](http://jmlr.org/papers/v18/15-613.html). The approach is based on conditional gradient methods from convex optimization. We provide several sample implementations of Algorithm 1 (see page 13). Given the well-structured nature of the problems solved in Algorithm 1, there are algorithmic improvements that can be made to the implementations here, but these serve as a good starting point.

We consider the regularized best-subsets estimator defined as 
```
minimize	0.5*\| y - X \beta \|_2^2 + \lambda \| \beta \|_q
subject to \| \beta \|_0  <= K,
```
where \lambda and K are two regularizers, and l_q is the l_1, l_2 or l_2^2 regularization. Our heuristic cycles through two sequences of parameters to build a regularization surface of near-optimal high-quality solutions.



### Implementations

The two implementations are as follows:

1. [Python]

We implement the Discrete First Order (DFO) algorithm and the Neighborhood Continuation heuristic respectively presented in Sections 2.2 and 2.3 of the paper. We also propose an MIO solver for the regularized best subset problem using Gurobi. The settings of the experiment is as described in the Experiment Section 5 of the paper

2. [Julia]


### Citation

If you would like to cite this work, please use the following citation for BCM17:

```
