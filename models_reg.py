import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Parameter
from torch.nn import Linear
from copy import deepcopy
from torch_geometric.nn.conv.sage_conv import SAGEConv
import types
import scipy
from torch_geometric.nn import GCNConv




class Model(nn.Module):
    def __init__(self, model_type, depth, nfeat, nhid, nout, dropout, *additional_params): 
        super(Model, self).__init__()

        self.model_type = model_type

        print(additional_params)
        # learning layers 
        self.input_linear = nn.Linear(nfeat, nhid)

        self.layers = nn.ModuleList()
        for d in range(depth):
            if model_type == 'gcn':
                self.layers.append(GCN_Layer(nhid))
            elif model_type == 'mlp':
                self.layers.append(Linear(nhid, nhid))
            elif model_type == 'sage':
                self.layers.append(SAGE_Layer(nhid))
           
        self.output_linear = nn.Linear(in_features=nhid, out_features=nout)

        # additional layers for training 
        self.norm = nn.BatchNorm1d(nhid)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, edge_index):

        x = self.input_linear(x)
        x = self.act(x)
        x = self.dropout(x)


        for layer in self.layers:
            if self.model_type == 'mlp':
                x = layer(x)
            else:
                x = layer(x, edge_index)
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.output_linear(x)
        return x


############ GCN ##############
class GCN_Layer(nn.Module):
    def __init__(self, dim):
        super(GCN_Layer, self).__init__()
        
        self.layer = GCNConv(dim, dim)

    def forward(self, x, edge_index):
        
        x = self.layer(x, edge_index)
        return x


########### GRAPHSAGE #############
class SAGE_Layer(nn.Module):
    def __init__(self, dim):
        super(SAGE_Layer, self).__init__()
        
        self.layer = SAGEConv(dim, dim, normalize=True)
        self.layer.aggr = 'mean'

    def forward(self, x, edge_index):
        
        x = self.layer(x, edge_index)
        return x

