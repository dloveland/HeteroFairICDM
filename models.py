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
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing, FAConv




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
            elif model_type == 'fagcn':
                self.layers.append(FAGCN_layer(nhid, eps=float(additional_params[0])))
            elif model_type == 'gcnii':
                self.layers.append(GCNII_Layer(nhid, d, alpha=float(additional_params[0]), theta=float(additional_params[1])))

        self.output_linear = nn.Linear(in_features=nhid, out_features=nout)

        # additional layers for training 
        self.norm = nn.BatchNorm1d(nhid)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, edge_index):

        x = self.input_linear(x)
        x = self.act(x)
        x = self.dropout(x)
 
        # residual connection back to initial features 
        if self.model_type in ['fagcn', 'gcnii']:
            x_0 = x.detach().clone() 

        for layer in self.layers:
            if self.model_type in ['fagcn', 'gcnii']:
                x = layer(x, x_0, edge_index)
            elif self.model_type == 'mlp':
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


########### GCNII #############

class GCNII_Layer(nn.Module):
    def __init__(self, dim, layer, alpha=0.1, theta=0.5, shared_weights=True):
        super().__init__()

        self.layer = GCN2Conv(dim, alpha, theta, layer + 1, shared_weights, normalize=False)

    def forward(self, x, x_0, edge_index):
        
        x = self.layer(x, x_0, edge_index)
        return x

########### FAGCN #############
class FAGCN_layer(torch.nn.Module):
    def __init__(self, dim, eps=0.1):
        super(FAGCN_layer, self).__init__()
        self.layer = FAConv(dim, eps)
 
    def forward(self, x, x_0, edge_index):
        x = self.layer(x, x_0, edge_index)
        return x
 

