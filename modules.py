import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
        
class deep_gcn(nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes, depth):
        super(deep_gcn, self).__init__()
        self.depth = depth

        self.conv1 = GCNConv(num_node_features, num_hidden)
        self.convs = nn.ModuleList([GCNConv(num_hidden, num_hidden) for _ in range(depth-1)])
        
        self.linear1 = nn.Linear(num_hidden, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_classes)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # first conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        for i in range(self.depth-1):

            # conv
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, p=0.1, training=self.training)


        # pooling
        x = global_mean_pool(x, data.batch)
        
        # linear layers
        x = torch.unsqueeze(x, dim=0) # unsqueeze for linear layer (simulate batch)
        x = self.linear1(x)
        #x = F.dropout(x)
        x = F.relu(x)

        x = self.linear2(x)
        
        # classification
        x = F.log_softmax(x, dim=2)
        x = x.view(*x.shape[1:])

        return x