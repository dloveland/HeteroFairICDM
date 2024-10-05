from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import os
import numpy as np
from tqdm import tqdm
from models_fairgnn import FairGNN
from models_nifty import NIFTY
import sys
import pickle 
import json
import itertools
import networkx as nx
from utils import fair_metric
import argparse 
from dataloader import load_dataset
from torch_geometric.nn.models import LINKX


        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='pokec-z, pokec-n, fb100-penn94, or tokolers')
    parser.add_argument('--model', type=str, default='nifty-gcn', help='model type')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--nclass', type=int, default=5, help='number of classes')
    parser.add_argument('--stratify', type=str2bool, default=True, help='stratify the dataset')
    parser.add_argument('--class_power', type=float, default=0.0, help='power for class stratification')
    parser.add_argument('--a', type=float, default=None)
    parser.add_argument('--b', type=float, default=None)

    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    base_fairness_model = args.model.split('-')[0]
    gnn_model = args.model.split('-')[1]

    save_folder = f'{args.dataset}_{args.model}_stratify_class_{args.class_power}_{args.seed}'

    if args.a and args.b:
        save_folder += f'_a_{args.a}_b_{args.b}'

    if os.path.exists(f'{base_fairness_model}_fairness_results/{save_folder}/iter_{0}.npy'):
        sys.exit()

    feats, labels, sens, adj, idx_train, idx_val, idx_test = load_dataset(args.dataset, device, stratify=True, \
                                                                                    seed=args.seed, class_power=args.class_power, \
                                                                                    a=args.a, b=args.b)  
    
    if base_fairness_model == 'nifty':
        if args.dataset == 'tolokers':
            sens_idx = -1
        elif args.dataset == 'pokec':
            sens_idx = 2
        else:
            sens_idx = 1

        model = NIFTY(
            adj,
            feats,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            sens_idx,
            num_hidden=16,
            num_proj_hidden=16,
            lr=0.001,
            weight_decay=1e-5,
            drop_edge_rate_1=0.1,
            drop_edge_rate_2=0.1,
            drop_feature_rate_1=0.1,
            drop_feature_rate_2=0.1,
            encoder=gnn_model,
            sim_coeff=0.6,
            nclass=args.nclass,
            device="cpu",
        )

        model.fit()

        (
            F1,
            SP,
            EO,
        ) = model.predict()


        print("F1: ", F1)
        print("SP: ", SP)
       
        

    elif base_fairness_model == 'fairgnn':

        # Initiate the model (with searched parameters).
        model = FairGNN(
            feats.shape[-1],
            model=args.model, 
            epoch=1000,
            nclass=args.nclass
        )
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, idx_train)


        # Evaluate the model.

        (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        ) = model.predict(idx_test)

        print("F1: ", F1)
        print("SP: ", SP)

    if not os.path.exists(f'{base_fairness_model}_fairness_results/{save_folder}'):
        os.makedirs(f'{base_fairness_model}_fairness_results/{save_folder}')   

    np.save(f'{base_fairness_model}_fairness_results/{save_folder}/iter_0.npy', np.array([F1, SP]))
