from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import os
import numpy as np
from tqdm import tqdm
from models_reg import *
import sys
import pickle 
import json
import itertools
import networkx as nx
from utils import fair_metric
import argparse 
from dataloader import load_dataset
from torch_geometric.nn.models import LINKX

class Trainer():
    def __init__(self, model_type, nclass, device, save_folder, weight_decay=5e-4, epochs=500):
        self.model_type = model_type
        self.nclass = nclass
        self.device = device
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.save_folder = save_folder
        
        if nclass == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        with open("params.json", "r") as param_file:
            params = json.load(param_file)

        possible_params = params[self.model_type]
        param_names = [] 
        param_vals = []
        for p in possible_params:
            param_names.append(list(p.keys())[0])
            param_vals.append(list(p.values())[0])

        self.param_names = param_names 
        self.param_vals = param_vals

        self.param_prod_list = list(itertools.product(*param_vals))

    def eval_model(self, model, features, labels, eval_idx, edge_index):
        
        model.eval()
        output = model(features, edge_index)
        if self.nclass == 1:
            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
        else:
            # Cross-Entropy
            preds = torch.argmax(output, axis=1)
        preds = preds[eval_idx].cpu().numpy()
        f1 = f1_score(labels[eval_idx].cpu().numpy(), preds, average='binary' if self.nclass == 1 else 'weighted')
        return f1, preds

    def train_model(self, model, features, labels, optimizer, idx_train, idx_val, edge_index, run_iter=0):

        best_f1 = -1 

        for epoch in tqdm(range(self.epochs)):
            model.train()
            optimizer.zero_grad()

            output = model(features, edge_index)

            if self.nclass == 1:
                # Binary Cross-Entropy  
                loss_train = self.criterion(output[idx_train].squeeze(), labels[idx_train].float())
            else:
                # Cross-Entropy
                loss_train = self.criterion(output[idx_train], labels[idx_train])
         
            loss_train.backward()
            optimizer.step()

            f1_val, _ = self.eval_model(model, features, labels, idx_val, edge_index)
            if f1_val > best_f1:
                best_f1 = f1_val
                torch.save(model.state_dict(), 'weights/{0}/iter_{1}.pt'.format(self.save_folder, run_iter))

    def hyperparam_search(self, features, labels, num_nodes, idx_train, idx_val, edge_index, nclass, run_iter=0):

        best_model = None 
        cross_model_f1 = -1
        best_params = None 

        results = {}
        # go through each parameter combination through grid search and init model
        for p in self.param_prod_list: 
            print(p)
            depth = p[0]
            hidden = p[1]
            dropout = p[2]
            lr = p[3]
        
            if len(p) > 3:
                additional_params = p[4:] 

            if self.model_type in ['gcn', 'sage', 'fagcn', 'gcnii', 'mlp']:
                model = Model(self.model_type, depth, features.shape[1], hidden, nclass, dropout, *additional_params)
            elif self.model_type == 'linkx':
                model = LINKX(num_nodes=num_nodes, in_channels=features.shape[1], hidden_channels=hidden, num_layers=depth, out_channels=nclass, dropout=dropout)
            else:
                raise ValueError('Model type not supported')
          
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
            model = model.to(self.device)

            self.train_model(model, features, labels, optimizer, idx_train, idx_val, edge_index, run_iter=run_iter)
                        
            # Get the best model over training for this param set 
            model.load_state_dict(torch.load('weights/{0}/iter_{1}.pt'.format(self.save_folder, run_iter)))

            f1_val, _ = self.eval_model(model, features, labels, idx_val, edge_index)
            results[p] = f1_val 

            if f1_val > cross_model_f1:
                cross_model_f1 = f1_val
                best_model = model 
                best_params = p 

            
        torch.save(best_model.state_dict(), 'weights/{0}/iter_{1}.pt'.format(self.save_folder, run_iter))
        with open('results/{0}/iter_{1}_hyper_param_opt.pkl'.format(self.save_folder, run_iter), 'wb') as f:
            pickle.dump(results, f)

        with open('results/{0}/iter_{1}_best_params.txt'.format(self.save_folder, run_iter), 'w') as f:
            for i, name in enumerate(self.param_names):
                f.write('{0},{1}\n'.format(name, best_params[i]))

        return best_model

        
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
    parser.add_argument('--dataset', type=str, default='fb100-cornell5', help='pokec-z, pokec-n, fb100-cornell5, fb100-penn94, or tokolers')
    parser.add_argument('--model', type=str, default='gcn', help='model type')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--nclass', type=int, default=5, help='number of classes')
    parser.add_argument('--stratify', type=str2bool, default=True, help='stratify the dataset')
    parser.add_argument('--class_power', type=float, default=0.0, help='power for class stratification')
    parser.add_argument('--a', type=float, default=None)
    parser.add_argument('--b', type=float, default=None)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    features, labels, sens, edge_index, idx_train, idx_val, idx_test = load_dataset(args.dataset, device, stratify=True, \
                                                                                    seed=args.seed, class_power=args.class_power, \
                                                                                    a=args.a, b=args.b)  
                    

    save_folder = f'{args.dataset}_{args.model}_stratify_class_{args.class_power}_{args.seed}'

    if args.a and args.b:
        save_folder += f'_a_{args.a}_b_{args.b}'

    trainer = Trainer(args.model, args.nclass, device, save_folder)
    num_nodes = features.shape[0]
    
    if not os.path.exists('weights/{0}'.format(save_folder)):
        os.makedirs('weights/{0}'.format(save_folder))
    if not os.path.exists('results/{0}'.format(save_folder)):
        os.makedirs('results/{0}'.format(save_folder))    
    run = 0 
    # check if results file already exists, if so, dont need to run 
    if not os.path.exists('results/{0}/iter_{1}.pkl'.format(save_folder, run)):
        # Perform hyper-parameter tuning for the model 
        model = trainer.hyperparam_search(features, labels, num_nodes, idx_train, idx_val, edge_index, args.nclass, run_iter=run)
    else:
        print('Already processed iter {0} for {1}'.format(run, args.model))
        sys.exit() 

    # evaluate the best model 
    f1_val, preds = trainer.eval_model(model, features, labels, idx_test, edge_index)

    parity, equality = fair_metric(preds, labels[idx_test].cpu().numpy(), sens[idx_test])

    results = {'demographic_party': parity,
                'f1': f1_val,
                'predictions': preds, 
                'labels': labels[idx_test].cpu().numpy(),
                'sens': sens[idx_test],
                'idx_test': idx_test 
                }

    with open(f'results/{save_folder}/iter_{run}.pkl', 'wb') as f:
        pickle.dump(results, f)

    
