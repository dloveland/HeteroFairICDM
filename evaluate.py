import os
import torch
from models_reg import *
from dataloader import load_dataset
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric
import networkx as nx
from torch_geometric.utils import convert, is_undirected, to_undirected, homophily
from torch_geometric.data import Data
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.nn.models import LINKX
import ot
import itertools 
import argparse
import sys
import pickle 
from utils import fair_metric_np
from scipy.stats import wasserstein_distance 
import argparse 

def get_hist(val, num_bins):

    # get binned values of r normalized
    hist, bin_edges = np.histogram(val, bins=num_bins, range=(0,1))
    hist = hist / float(hist.sum())

    return hist, bin_edges

def get_local_homophily(edge_index, metric, radius=1):
    all_local_homophily = []

    num_nodes = len(metric)
    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        neighbor_metrics = metric[neighbors]
        same = (neighbor_metrics == metric[node]).sum()
        local_homophily = same / float(len(neighbor_metrics))
        all_local_homophily.append(local_homophily)
    
    return np.array(all_local_homophily)


def emd(X, Y, num_bins=5):

    wass = wasserstein_distance(X, Y)
    
    return wass  

class Evaluator():
    def __init__(self, model_type, save_path, nclass, num_nodes, num_feats, iter=0):
        self.model_type = model_type
        self.save_path = save_path
        self.nclass = nclass
        self.weights_path = os.path.join('weights', self.save_path, f'iter_{iter}.pt')
        self.params_path = os.path.join('results', self.save_path, f'iter_{iter}_best_params.txt')

        with open(self.params_path) as param_file:
            params = param_file.readlines()
            params = [p.strip().split(',')[1] for p in params]
        
        depth = int(params[0])
        hidden = int(params[1])
        dropout = float(params[2])
        try:
            lr = params[3]
        except:
            lr = 0.001
            params.append(lr)

        if len(params) > 3:
            additional_params = params[4:] 
            additionals_params = [float(p) for p in additional_params]


        if self.model_type in ['gcn', 'sage', 'fagcn', 'gcnii', 'mlp']:
            model = Model(self.model_type, depth, num_feats, hidden, nclass, dropout, *additional_params)
        elif self.model_type == 'linkx':
            model = LINKX(num_nodes=num_nodes, in_channels=num_feats, hidden_channels=hidden, num_layers=depth, out_channels=nclass, dropout=dropout)
        elif self.model_type == 'sgc':
            model = SGC(nfeat=num_feats, nclass=nclass, depth=depth)
        else:
            raise ValueError('Model type not supported')

        model.load_state_dict(torch.load(self.weights_path))
        self.model = model 

    def eval_model(self, features, labels, sens, eval_idx, edge_index):

        self.model.eval()
        output = self.model(features, edge_index)
        if self.nclass == 1:
            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
        else:
            # Cross-Entropy
            preds = torch.argmax(output, axis=1)
        test_preds = preds[eval_idx].cpu().numpy()
        test_labels = labels[eval_idx].cpu().numpy()
        test_sens = sens[eval_idx].cpu().numpy()

        #f1 = f1_score(test_labels, test_preds, average='binary' if self.nclass == 1 else 'micro')
        f1 = (test_labels == test_preds).sum() / float(len(test_labels))
        parity, equality = fair_metric_np(test_preds, test_labels, test_sens, multi=True if nclass != 1 else False)

        return f1, parity

def evaluate_data(model, dataset, a, b, class_power, seed, nclass, device):
    
    features, labels, sens, edge_index, idx_train, idx_val, idx_test = load_dataset(dataset, device, stratify=True, \
                                                                                    seed=seed, class_power=class_power, \
                                                                                    a=a, b=b)  

    model_save_folder = f'{dataset}_{model}_stratify_class_{class_power}_{seed}'
    if a and b:
        model_save_folder += f'_a_{a}_b_{b}'

    try:
        evaluator = Evaluator(model, model_save_folder, nclass, features.shape[0], features.shape[1], iter=0)
    except Exception as e:
        print(e)
        return None, None
                                  
    f1, parity = evaluator.eval_model(features, labels, sens, idx_test, edge_index)

    local_homophily = get_local_homophily(edge_index, labels)
    train_local_homophily = local_homophily[idx_train]
    test_local_homophily = local_homophily[idx_test]
    
    train_hist = get_hist(train_local_homophily, 5)[0]
    test_hist = get_hist(test_local_homophily, 5)[0]
    
    dist = np.round(emd(train_hist, test_hist, num_bins=5), 3)

    values, counts = np.unique(labels[idx_train], return_counts=True)
    return f1, parity, dist, counts

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
     
    seeds = [123, 246, 492]
    shape_params = [None, (3.0, 10.0), (3.0,5.0), (5.0,3.0), (10.0,3.0)]
    gammas = [0.0, 1.0, 3.0]

    datasets = ['fb100-penn94', 'tolokers', 'pokec']

    results = {}

    if args.model in ['gcn', 'sage', 'mlp', 'linkx']:
        for model in [args.model]:
            results[model] = {}
            for dataset in datasets:  
                print(model, dataset)
                sys.stdout.flush()
                results[model][dataset] = {}

                if dataset in ['tolokers']:
                    nclass = 1
                elif dataset in ['pokec', 'fb100-penn94', 'fb100-cornell5']:
                    nclass = 5

                for shape_param in shape_params:  
                    results[model][dataset][shape_param] = {} 

                    if shape_param is None:
                        a = None
                        b = None
                    else: 
                        a = shape_param[0]
                        b = shape_param[1]

                    for seed in seeds: 
                        results[model][dataset][shape_param][seed] = {} 
                        f1s = []
                        parities = []
                        dists = []
                        counts = []
                        for class_power in gammas:       
                        
                            device = torch.device("cpu")

                            f1, parity, dist, count = evaluate_data(model, dataset, a, b, class_power, seed, nclass, device)
                            if f1 is not None:
                                f1s.append(f1)
                                parities.append(parity)
                                dists.append(dist)
                                counts.append(count)

                        results[model][dataset][shape_param][seed]['f1'] = f1s 
                        results[model][dataset][shape_param][seed]['parity'] = parities
                        results[model][dataset][shape_param][seed]['dists'] = dists
                        results[model][dataset][shape_param][seed]['counts'] = counts
                    
        with open(f'{args.model}_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    else: 
        # assume srtucture as fair-%
        model_name = args.model.split('-')[1]

        for model in [model_name]:
            results[model] = {}
            for dataset in datasets:  
                sys.stdout.flush()
                results[model][dataset] = {}

                if dataset in ['tolokers']:
                    nclass = 1
                elif dataset in ['pokec', 'fb100-penn94', 'fb100-cornell5']:
                    nclass = 5

                for shape_param in shape_params:  
                    results[model][dataset][shape_param] = {} 

                    if shape_param is None:
                        a = None
                        b = None
                    else: 
                        a = shape_param[0]
                        b = shape_param[1]

                    for seed in seeds: 
                        results[model][dataset][shape_param][seed] = {} 
                        f1s = []
                        parities = []
                        dists = []
                        counts = []
                        for class_power in gammas:       
                            device = torch.device("cpu")

                            save_folder = f'{dataset}_{model}_stratify_class_{class_power}_{seed}'

                            if a and b:
                                save_folder += f'_a_{a}_b_{b}'

                            data = np.load(f'nifty_fairness_results/{save_folder}/iter_{0}.npy')
                            if data is not None:
                                f1s.append(data[0])
                                parities.append(data[1])
                                
                        results[model][dataset][shape_param][seed]['f1'] = f1s 
                        results[model][dataset][shape_param][seed]['parity'] = parities
            
        with open(f'{args.model}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
