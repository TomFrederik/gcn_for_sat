import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class gcn(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes):
        super(gcn, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_hidden)
        self.linear1 = torch.nn.Linear(num_hidden, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden, num_classes)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # first conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        
        # second conv
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x)

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
        
