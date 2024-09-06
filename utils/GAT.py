# import packages
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# define GAT
class GAT(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 heads,
                 num_features,):
        super().__init__()
        self.conv1 = GATConv(num_features,
                             hidden_channels,
                             heads)
        self.conv2 = GATConv(hidden_channels * heads,
                             num_classes,
                             heads = 1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x