import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels1=256, hidden_channels2=128, out_channels=2, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels1)
        self.conv2 = SAGEConv(hidden_channels1, hidden_channels2)
        self.conv3 = SAGEConv(hidden_channels2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels1=256, hidden_channels2=128, out_channels=2, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels1=256, hidden_channels2=128, out_channels=2, dropout=0.3):
        super(GATModel, self).__init__()
        # Using heads=1 for simplicity, can be adjusted
        self.conv1 = GATConv(in_channels, hidden_channels1, heads=1)
        self.conv2 = GATConv(hidden_channels1, hidden_channels2, heads=1)
        self.conv3 = GATConv(hidden_channels2, out_channels, heads=1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x
