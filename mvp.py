import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from modules import gcn
import time
import matplotlib.pyplot as plt


def main(config):

    # convert to dict
    config = vars(config)

    # set random seed
    np.random.seed(config['seed'])

    # get path of code file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = file_dir + config['data_path']
    plot_path = file_dir + '/plots/'
    model_path = file_dir + '/models/'
    print('Data path is ' + data_path)
    
    # get list of files
    (_, _, filenames) = os.walk(data_path).__next__()
    edge_files = [file for file in filenames if file[:5] == 'edges']
    feature_files = [file for file in filenames if file[:8] == 'features']
    print('Preparing data..')

    # load data
    # each file contains a list of two lists of tensors that represent
    # the edges and features of graphs respectively.
    edges = [[], []]
    features = [[], []]
    for i in range(len(edge_files)):
        edge_data = torch.load(data_path+edge_files[i])
        edges[0].extend(edge_data[0])
        edges[1].extend(edge_data[1])
        features_data = torch.load(data_path+feature_files[i])
        features[0].extend(features_data[0])
        features[1].extend(features_data[1])
    
    # sample targets
    targets = np.random.choice([0, 1], size=len(edges[0])) # 0 means unsat, 1 means sat

    # use targets as mask to get formulas
    # for now only use the first file
    X_graph = []
    for i in range(len(edges[0])):
        X_graph.append([edges[targets[i]].pop(0), features[targets[i]].pop(0)])

    # clear from RAM
    edges.clear()
    features.clear()

    #####
    # train test split
    #####

    split_idx = 8000

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
    print('Preparing model...')
    
    num_node_features = 40 # need the same feature length everywhere
    num_hidden = 50
    num_classes = 2
    lr = 2e-3
    weight_decay = 1e-10
    max_epochs = 400
    batch_size = 50

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    

    model = gcn(num_node_features=num_node_features, num_hidden=num_hidden, num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = torch.nn.NLLLoss() #neg log likelihood loss

    ####
    # train
    ####
    print('Training..')
    # training mode
    model.train()

    losses = []
    accs = []

    # trainings loop
    for epoch in range(max_epochs):
        
        epoch_loss = 0
        epoch_acc = 0
        steps = 0

        for i in range(0,len(train_idcs)-batch_size,batch_size):
            steps += 1            
        
            y_batch = torch.from_numpy(targets[i:i+batch_size]).long().to(device)
            x_batch = Batch.from_data_list([Data(x=X_graph[train_idcs[i+k]][1], edge_index=X_graph[train_idcs[i+k]][0]) for k in range(batch_size)]).to(device)
            
            # forward pass
            out = model(x_batch)
            
            # compute accuracy
            probs = torch.exp(out) 
            indices = torch.argmax(probs, dim=1)
            correct = (y_batch.eq(indices.long())).sum()
            accuracy = correct.item() / y_batch.shape[0]

            # backward pass
            optimizer.zero_grad()
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            # logging
            epoch_loss += loss.item()
            epoch_acc += accuracy
        # more logging
        epoch_acc /= steps
        epoch_loss /= steps
        accs.append(epoch_acc)    
        losses.append(epoch_loss)

        # adapt learning rate
        scheduler.step()

        # monitoring
        print('In epoch {0:4d} the average training loss is {1:2.5f}'.format(epoch, epoch_loss))
        print('In epoch {0:4d} the average training acc is {1:2.5f}'.format(epoch, epoch_acc))

    # save plots
    plt.figure(1)
    plt.plot(np.arange(max_epochs), losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(plot_path + 'train_loss_' + str(time.time())+ '.pdf')
    
    plt.figure(2)
    plt.plot(np.arange(max_epochs), accs)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(plot_path + 'train_acc_' + str(time.time())+ '.pdf')

    ###
    # save model
    ###
    torch.save(model.state_dict(), model_path + 'run_' + str(time.time()) + '.pt')

if __name__ == "__main__":
     # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seed', type=int, default=314159, help='Random seed')
    parser.add_argument('--data_path', type=str, default='/data/', help='Path to save the datasets')    
    
    config = parser.parse_args()

    # run main program
    main(config)
