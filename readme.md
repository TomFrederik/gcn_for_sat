## Readme
This repository implements a graph convolutional network to solve SAT formulas with 10 - 40 variables.

The idea of how to generate the formulas is taken from Selsam et al. (2019) 'Learning a SAT solver from single-bit supervision'

## The SAT distribution
As in the paper, the formulas are sampled as follows:

For each formula we draw the number of variables n from the uniform distribution between 10 and 40, U(10,40).

While the formula is still satisfiable, generate a clause as follows:
Draw the number of literals k from the distribution 1 + Bernoulli(p=0.7) + Geo (p=0.4). If, by chance, k > n, then set k = n.

Now draw k variables without replacement and uniform probaility from the n variables. Finally negate each of the k variables with independent probability 0.5.

Test whether the formula is still satisfiable. If yes, generate another clause, if no, then save the formula and the formula with a random literal negated.

The negated literal means that the formula is satisfiable.

We repeat this a hundred thousand times and have 100,000 pairs of sat formulas, one satisfiable, the other one not.

At test time we will sample randomly whether to select the sat or the unsat one.

## The graph representation
In the paper, the authors did not mention which embedding they used for their node features. I suspect they mostly relied on the topology of the graphs so that the features did not really matter as much, since they used a message passing scheme.

I on the other hand wanted to work with the features explicitly so I opted for a different topology. Each variable is assigned its own node with unit features of lenght 40 (max number of variables). Each clause has its own node.
If a variable is in a clause (negative or positive), we define an edge between the two corresponding nodes. 

The clause node features are 1 x 40 vectors with entries of 1 (positive variableis in clause), -1 (negative) or 0 (variable not in clause)

## The network
I use the pytorch geometric package to do the heavy lifting of the graph convolutions.

