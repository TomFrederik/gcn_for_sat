import scipy.stats as stats
import subprocess as sub
import numpy as np
import argparse
import pysat.solvers as solv
from pysat.solvers import Glucose3
import time
import os

def get_clause(n, dists):
    '''
    input:
    n - number of variables to choose from
    dists - a tuple of distributions over integers to sample k from (samples will be added)

    returns:
    clause - a new clause
    '''
    # sample k
    k = sum([int(dist.rvs(size=1)) for dist in dists])
    if k > n:
        k = n

    # sample a new clause
    clause = sample_clause(k, n)

    return clause

def sample_clause(k, n):
    '''
    input:
    k - number of literals in the clause
    n - number of variables to choose from

    returns:
    clause - a list of integers, where negative means negation
    '''
    # list of variables
    vars = np.arange(1, n+1, 1)

    # sample k variables without replacement
    choice = np.random.choice(a=vars, size=k, replace=False)

    # negate each with probability 50%
    mask = np.random.choice(a=[-1,1], size=k, replace=True)
    clause = list(choice * mask)
    clause = [int(x) for x in clause]

    return clause


def main(config):

    # convert to dict
    config = vars(config)

    # set random seed
    np.random.seed(config['seed'])

    # get path of code file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = file_dir + config['data_path']
    print('Data path is ' + data_path)
    
    # instantiate distributions for nbr of literals per clause
    geo = stats.geom(config['geo_p'])
    bern = stats.bernoulli(config['bern_p'])
    one = stats.uniform(loc=1, scale=0) # always returns 1
    dists = (geo, bern, one)

    # instantiate distro for nbr of variables for a formula
    nbr_vars = stats.uniform(loc=config['range_nbr_vars'][0], scale=config['range_nbr_vars'][1]-config['range_nbr_vars'][0])

    # list of formulas
    dataset = []

    # number of literals per clause 
    for i in range(config['nbr_data']):
        
        phi = Glucose3() # instantiate phi as 0 clauses
        list_phi = [] # phi as a list for saving and inspection purposes
        n = int(nbr_vars.rvs(size=1)) #nbr of vars to use in this formula
        
        sat = True

        while sat: 
            
            # get new clause
            new_clause = get_clause(n, dists)

            # add clause
            phi.add_clause(new_clause)
            list_phi.append(new_clause)

            #solvable?
            sat = phi.solve()
        
        # revert a random literal in a random clause
        # to obtain a satisfiable formula
        clause_idx = np.random.choice(range(len(list_phi)), size=1)[0]
        lit_idx = np.random.choice(range(len(list_phi[clause_idx])), size=1)[0]
        sat_list_phi = list_phi.copy()
        sat_list_phi[clause_idx][lit_idx] *= -1
        
        # append 2-tuple of [sat, unsat] to dataset
        dataset.append([list_phi, sat_list_phi])

        if i % 100 == 0:
            print('{} formulas generated in total.'.format(i))

    # save dataset
    np.save(data_path + 'data_' + str(int(time.time())) + '.npy', dataset)


if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--geo_p', type=float, default=0.4, help='Parameter for geometric distribution')
    parser.add_argument('--bern_p', type=float, default=0.7, help='Parameter for bernoulli distribution')
    parser.add_argument('--nbr_data', type=int, default=10000, help='Number of datapoints to be generated')
    parser.add_argument('--seed', type=int, default=314159, help='Random seed')
    parser.add_argument('--data_path', type=str, default='/data/', help='Path to save the datasets')
    parser.add_argument('--range_nbr_vars', type=list, default=[10, 40], help='Range of number of random variables')
    
    
    config = parser.parse_args()

    # run main program
    main(config)
