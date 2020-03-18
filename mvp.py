import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data
from modules import gcn

def make_graph(formula):
    '''
    converts a sat formula to a graph
    input: A sat formula as list of clauses (lists)
    returns: the edge indices, a 2 X E tensor, and the feature vectors for each node

    Each variable and clause is represented by a node.
    If a variable is (negatively or positively) part of a clause, there is an edge between the two
    The features for each variable is a 1 x N vector of 1s
    The feature for each clause is a 1 x N vector encoding whether each variable is positive, negative
    or not contained in this clause
    '''

    # disregard sign of literal for edges
    abs_clauses = [np.abs(clause) for clause in formula]

    n = np.max([np.max(clause) for clause in abs_clauses]) # nbr of variables

    
    edge_indices = []
    features = []
    
    # features for nodes are just 1, need to have same number across all graphs
    features.append(torch.ones((n,40), dtype=torch.float))
    
    for i in range(len(abs_clauses)):
        feature = torch.zeros((1, 40), dtype=torch.float)
        for j in range(len(abs_clauses[i])):
            # undirected edge between clause i and variable at position j in clause i
            edge_indices.append(torch.tensor([[abs_clauses[i][j]-1, n+i]], dtype=torch.long))
            edge_indices.append(torch.tensor([[n+i, abs_clauses[i][j]-1]], dtype=torch.long))

            # feature +1 if literal is positive, -1 if negative, 0 if not there
            feature[:,abs_clauses[i][j]-1] = torch.sign(torch.tensor(formula[i][j], dtype=torch.float)).float()
            features.append(feature)

    # make one tensor
    edge_index = torch.cat(edge_indices, dim=0)
    features = torch.cat(features, dim=0)

    # transpose to bring into format for torch_geometric
    edge_index = edge_index.t().contiguous()

    return edge_index, features


def get_graphs(formulas):
    '''
    input: list of sat formulas
    returns: list of graphs, represented by tuples [edge_indices, features]
    '''
    graphs = []
    for formula in formulas:
        edge_index, features = make_graph(formula)
        graphs.append([[edge_index, features]])

    return graphs

def main(config):

    # convert to dict
    config = vars(config)

    # set random seed
    np.random.seed(config['seed'])

    # get path of code file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = file_dir + config['data_path']
    print('Data path is ' + data_path)
    
    # get list of files
    (_, _, filenames) = os.walk(data_path).__next__()
    
    # load data
    data = [np.load(data_path+file, allow_pickle=True) for file in filenames]
    
    # sample targets
    targets = np.random.choice([0, 1], size=len(data[0])) # 0 means unsat, 1 means sat

    # use targets as mask to get formulas
    X = data[0][np.arange(0,len(data[0])), targets]

    # transform into graph
    # i.e. edge indices and feature vectors
    X_graph = get_graphs(X)


    #####
    # train test split
    #####

    split_idx = 7000

    shuffle_idcs = np.arange(0,len(X_graph))
    np.random.shuffle(shuffle_idcs)

    train_idcs = shuffle_idcs[:split_idx]
    test_idcs = shuffle_idcs[split_idx:]

    #X_train = X_graph[shuffle_idcs[:split_idx]]
    #X_test = X_graph[shuffle_idcs[split_idx:]]

    y_train = targets[shuffle_idcs[:split_idx]]
    y_test = targets[shuffle_idcs[split_idx:]]


    ####
    # setting up
    ####
    num_node_features = 40 # need the same feature length everywhere
    num_classes = 2
    lr = 1e-2
    weight_decay = 5e-4
    max_epochs = 10

    if torch.cuda.is_available():
        device = 'cuda0'
    else:
        device = 'cpu'
    

    model = gcn(num_node_features=num_node_features, num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    ####
    # train
    ####

    # training mode
    model.train()

    # trainings loop
    for epoch in range(max_epochs):
        for i in range(len(train_idcs)):
            data = Data(x=X[train_idcs[i]][1], edge_index=X[train_idcs[i]][0]).to(device)
            optimizer.zero_grad()
            out = model(data)
            if y_train[i] == 0:
                target = torch.tensor([1,0], dtype=torch.float, device=device)
            else:
                target = torch.tensor([0,1], dtype=torch.float, device=device)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('In epoch {0:3.0d} the average training loss is {1:2.5f}'.format(epoch, epoch_loss/len(train_idcs)))

if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seed', type=int, default=314159, help='Random seed')
    parser.add_argument('--data_path', type=str, default='/data/', help='Path to save the datasets')    
    
    config = parser.parse_args()

    # run main program
    main(config)
