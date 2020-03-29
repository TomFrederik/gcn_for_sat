## Readme
This repository implements a graph convolutional network to solve SAT formulas with 10 - 40 variables.

The idea of how to generate the formulas is taken from Selsam et al. (2019) 'Learning a SAT solver from single-bit supervision'

## The SAT distribution
As in the paper, the formulas are sampled as follows:

1. For each formula we draw the number of variables n from the uniform distribution between 10 and 40, U(10,40).

2. While the formula is still satisfiable, generate a clause as follows:

  Draw the number of literals k from the distribution 1 + Bernoulli(p=0.7) + Geo(p=0.4). 

  If, by chance, k > n, then set k = n.

3. Now draw k variables without replacement and uniform probaility from the n variables. 

4. Finally negate each of the k variables with independent probability 0.5.

5. Test whether the formula is still satisfiable. If yes, generate another clause, if no, then save the formula and the formula with a random literal negated.

  The negated literal means that the formula is satisfiable.

6. We repeat this fifty thousand times and have 50,000 pairs of sat formulas, one satisfiable, the other one not.

At test time we will sample randomly whether to select the sat or the unsat formula.

## The graph representation
In the paper, the authors did not mention which embedding they used for their node features. I suspect they mostly relied on the topology of the graphs so that the features did not really matter as much.

I on the other hand wanted to work with the features explicitly so I opted for a different topology. Each variable is assigned its own node with unit features of lenght 40 (max number of variables). Each clause has its own node.
If a variable is in a clause (negative or positive), we define an edge between the two corresponding nodes. 

The clause node features are 1 x 40 vectors with entries of 1 (positive variable is in clause), -1 (negative) or 0 (variable not in clause)

## The network
I use the pytorch geometric package to do the heavy lifting of the graph convolutions.

The network is comprised of two graph convolutional layers and two linear layers for classification.
After each layer ReLU and dropout is employed, except for the last layer after which we just use log_softmax to assign probabilities to the two classes (sat, unsat).

As a loss, the negative log-likelihood loss is used.

## Best Hyperparameters
weight decay = 1e-10

lr = 2e-3

step_decay = factor 10 each 50 epochs

max_epochs = 100

batch_size = 100

## Results
# All data at once
The model does not learn the test set, rather simply learns the training set, leading to a significant overfitting. I do not know whether this is due  to the different topology, which might make the problem harder, or due to an insufficient amount of training data. In the paper, the authors claim to have used millions of training samples compared to my 50.000 which might explain it.
<object data="https://github.com/TomFrederik/gcn_for_sat/blob/master/plots/acc_1585216030.118168.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/TomFrederik/gcn_for_sat/blob/master/plots/acc_1585216030.118168.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/TomFrederik/gcn_for_sat/blob/master/plots/acc_1585216030.118168.pdf">Download PDF</a>.</p>
    </embed>
</object>

# Switch files
In an attempt to avoid overfitting, and to handle larger amounts of data, I switch from having one big training set to training/evaluating on one set, then, once an early stopping criterion is reached, I switch to the next set. 



