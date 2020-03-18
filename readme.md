This repository implements a graph convolutional network to solve SAT formulas with 10 - 40 variables.

The idea of how to generate the formulas is taken from Selsam et al. (2019) 'Learning a SAT solver from single-bit supervision'

## The SAT distribution
As in the paper, the formulas are sampled as follows:

For each formula we draw the number of variables n from the uniform distribution between 10 and 40, U(10,40).

While the formula is still satisfiable, generate a clause as follows:
Draw the number of literals k from the distribution 1 + Bernoulli(p=0.7) + Geo (p=0.4). If, by chance, k > n, then set k = n.

Now draw k variables without replacement and uniform probaility from the n variables. Finally negate each of the k variables with independent probability 0.5.
