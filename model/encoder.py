import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from torch_geometric.nn import GCNConv, RGCNConv, FastRGCNConv
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
from torch_geometric.data import HeteroData

from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GraphConv, GATConv, GATv2Conv, HGTConv, Linear

from model.layer import WeightedRGCNConv
from model.regressor import Regressor


class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    

class RGCNLayerWithSkip(torch.nn.Module):
    def __init__(self, skip_in_channels, in_channels, out_channels, num_relations):
        super(RGCNLayerWithSkip, self).__init__()
        self.conv = WeightedRGCNConv(in_channels, out_channels, num_relations)
        self.skip_projection = torch.nn.Linear(skip_in_channels, out_channels)

    def forward(self, x, h, edge_index, edge_type, edge_weight=None):
        skip = self.skip_projection(x)
        h = self.conv(h, edge_index, edge_type, edge_weight)
        h = h + skip
        return h


class WRGCN(torch.nn.Module):
    def __init__(self, input_dim, dim, num_layers=2, dropout=0.0, num_relations=3, projection=False):
        super().__init__()

        self.dim = dim

        self.gnn_dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.num_relations = num_relations

        if projection:
            self.projection = Projection(input_dim, dim)

        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            if l == 0 and not projection:
                self.convs.append(RGCNLayerWithSkip(input_dim, input_dim, dim, num_relations))
            elif not projection:
                self.convs.append(RGCNLayerWithSkip(input_dim, dim, dim, num_relations))
            else:
                self.convs.append(RGCNLayerWithSkip(dim, dim, dim, num_relations))


    def forward(self, x, edge_index, edge_type, edge_weight=None):
        if hasattr(self, 'projection'):
            x = self.projection(x)

        h = x
        for conv in self.convs:
            h = self.gnn_dropout(h)
            h = conv(x, h, edge_index, edge_type, edge_weight)

        return h

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()


