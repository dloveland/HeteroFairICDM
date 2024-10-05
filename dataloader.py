import contextlib
import pandas
import os
from pathlib import Path
import os
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import scipy
import sys
from torch_geometric.utils import convert, is_undirected, to_undirected, homophily, subgraph
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import networkx as nx
import collections
import torch_geometric.transforms as T
from collections import Counter

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

def get_local_homophily(edge_index, metric, radius=1):

    all_local_homophily = []
    num_neighs = []

    num_nodes = len(metric)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        neighbor_metrics = metric[neighbors]
        num_neighs.append(len(neighbors))
      
        same = (neighbor_metrics == metric[node]).sum()
        local_homophily = same / float(len(neighbor_metrics))
       
        all_local_homophily.append(local_homophily)
    
    return np.array(all_local_homophily), np.array(num_neighs)

def get_global_homophily(edge_index, metric):

    global_homophily = homophily(edge_index, metric, method='edge')
    return global_homophily

def stratify_idx(local_homophily_labels, distr, inv_distr, seed, num_bins=5, train_val_split=(0.8, 0.2)):

    train_percents = distr / (distr + inv_distr)
    test_percents = inv_distr / (distr + inv_distr)

    idx_train = []
    idx_val = []
    idx_test = []
    total = 0
    for label_bin in range(num_bins):
            
        train_percent = train_percents[label_bin] * train_val_split[0]
        val_percent = train_percents[label_bin] * train_val_split[1]
        test_percent = test_percents[label_bin]

        lower_label = (1/num_bins) * label_bin
        upper_label = (1/num_bins) * (label_bin + 1)
        if upper_label == 1:
            upper_label += 0.01
        idx = np.where((local_homophily_labels >= lower_label) & (local_homophily_labels < upper_label))[0]
        total += len(idx)

        np.random.shuffle(idx)

        idx_train.extend(idx[:int(len(idx)*train_percent)])
        idx_val.extend(idx[int(len(idx)*train_percent):int(len(idx)*(train_percent + val_percent))])
        idx_test.extend(idx[int(len(idx)*(train_percent + val_percent)):int(len(idx)*(train_percent + val_percent + test_percent))])

    return idx_train, idx_val, idx_test
    

def load_dataset(dataset, device, stratify=False, seed=None, labeled_subgraph=True, in_mem=True, class_power=1.0, a=None, b=None):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not in_mem:
        if 'pokec' in dataset:
            
            df = pd.read_csv('real_datasets/pokec/soc-pokec-profiles.csv', sep=',', header=None)
            
            # turn these all into a big list of strings
            col_names = ['user_id', 'public', 'completion_percentage', 'gender', 'region', 'last_login', 'registration', 'AGE', 'body', 'I_am_working_in_field']
            df.columns = col_names

            # sort df by user_id col
            df = df.sort_values(by='user_id')
            print(df.head())

            labels = df['I_am_working_in_field'].values
            # drop unnecessary cols and label 
            df = df.drop(columns=['user_id', 'body', 'I_am_working_in_field'])

            # TODO CLEAN FEATURES 
            features = df.to_numpy()
            
            sens = df['gender'].values
            sens = np.nan_to_num(sens, nan=-1)

            labels = ['missing' if x is np.nan else x for x in labels]
        
            job_types = dict(Counter(labels))
            job_keys = np.array(list(job_types.keys()))
            job_counts = list(job_types.values())

            # get indices of top k most common labels, remove first idx as its the missing label 
            k = 10
            top_k_labels = job_keys[np.argsort(job_counts)[-k-2:-2]]

            # get indices of data points with labels in top k
            label_idx = np.where(np.isin(labels, top_k_labels))[0] 
            # get indices in sens that are non-zero
    
            sens_idx = np.where(sens >= 0)[0]
            # get intersection of label_idx and sens_idx
            label_idx = np.intersect1d(label_idx, sens_idx)

            # build graph
            edges_unordered = np.genfromtxt('real_datasets/pokec/soc-pokec-relationships.txt', dtype=int).T
            edge_index = torch.Tensor(edges_unordered).long() - 1
            
            if labeled_subgraph:
                
                # get the subgraph of the labeled nodes 
                edge_index = subgraph(torch.Tensor(label_idx).long(), edge_index, num_nodes=len(labels), relabel_nodes=True)[0]
                
                # get the features, labels, and sens of the unique nodes
                features = features[label_idx]
                labels = np.array(labels)[label_idx]
                #### PROCESS FEATURES 
                # 1 is percent complete 
                # 3 is region
                # 4 is last login
                # 5 is registration 
                # -1 is age 
                # one hoe encode column 3

                # divide percent complete and age by 100 to make it between 0 and 1
                features[:, 1] = features[:, 1] / 100
                features[:, -1] = features[:, -1] / 100

                # one hot encode the region 
                
                unique_region, count_region = np.unique(features[:, 3], return_counts=True)
                one_hot = np.zeros((len(features), len(unique_region)))
                for r in range(len(features)):
                    curr_reg = features[r, 3]
                    one_hot[r, np.where(unique_region == curr_reg)] = 1
                
                # convert last login column from datetime to seconds
                features[:, 4] = pd.to_datetime(features[:, 4]).astype('int64') // 10**9
                
                # convert registration column to seconds
                features[:, 5] = pd.to_datetime(features[:, 5]).astype('int64') // 10**9
                # subtract min registration time, i.e. start of site from both date-time cols

                # get datetime in seconds of the date May 28th 2012, when the data was scraped, and scale relative to this time and min registration time 
                max_seconds = (pd.to_datetime(['2012-05-28 00:00:00.0']).astype('int64') // 10**9).to_numpy()[0]
                features[:, 4] = (features[:, 4] - np.min(features[:, 5])) / (max_seconds - np.min(features[:, 5]))
                features[:, 5] = (features[:, 5] - np.min(features[:, 5])) / (max_seconds - np.min(features[:, 5]))

                # append one hot encoded regions and drop the original region column 
                features = np.concatenate((features, one_hot), axis=1)
                features = np.delete(features, 3, axis=1)

                # convert first columns to float
                features = features.astype('float32')
                
                features = torch.Tensor(features).to(device)

            labels = np.array(labels)
            # map labels to unique value between 0 and number of unique labels
            label_map = {label: i for i, label in enumerate(np.unique(labels))}
            labels = torch.LongTensor([label_map[label.item()] for label in labels])
            sens = torch.Tensor(sens[label_idx])
        
            data = Data(edge_index=edge_index, num_nodes=features.shape[0], x=features, y=labels, sens=sens)

            # get largest connected component of data
            data = T.LargestConnectedComponents()(data)

            edge_index = data.edge_index
            if labeled_subgraph:
                features = data.x.to(device)
                labels = data.y
                sens = data.sens
            labels = data.y
            sens = data.sens

            torch.save({'features': features, 'labels': labels, 'sens': sens, 'edge_index': edge_index}, 'real_datasets/pokec/pokec_processed.pt')

        elif 'fb100' in dataset:
            # columns are: student/faculty, gender, major,
            #              second major/minor, dorm/house, year/ high school
            # 0 denotes missing entry

           if dataset == 'fb100-penn94':
                dataset = 'Penn94'

            data = scipy.io.loadmat(f'real_datasets/facebook/{dataset}.mat')

            edge_index = torch.Tensor(np.array(data['A'].nonzero())).long()
            feature_labels = data['local_info']

            feature_idxs = range(feature_labels.shape[1])
            sens_attr = 1
            label = 2

            # remove the label from dataset
            feature_idxs = np.delete(feature_idxs, [label])

            # additionally remove the last feature, which high school 
            features = feature_labels[:, feature_idxs][:, :-1].astype('int32')
    
            labels = feature_labels[:, label].astype('int32')

            # get unique labels and counts 
            unique_labels = np.unique(labels, return_counts=True)

            counts = unique_labels[1]
            
            # get indices of top k most common labels
            k = 5
            # remove the last one as it is missing 
            top_k_labels = unique_labels[0][np.argsort(counts)[-k-1:-1]]
            
            # get indices of data points with labels in top k
            label_idx = np.where(np.isin(labels, top_k_labels))[0] 
            sens = feature_labels[:, sens_attr].astype('int32')
            # get indices in sens that are non-zero
            sens_idx = np.where(sens > 0)[0]
            # get intersection of label_idx and sens_idx
            label_idx = np.intersect1d(label_idx, sens_idx)

            
            if labeled_subgraph:
                
                # get the subgraph of the labeled nodes 
                edge_index = subgraph(torch.Tensor(label_idx).long(), edge_index, num_nodes=len(labels), relabel_nodes=True)[0]
                
                # get the features, labels, and sens of the unique nodes
                features = torch.Tensor(features[label_idx])
                labels = torch.LongTensor(labels[label_idx])
                # map labels to unique value between 0 and number of unique labels
                label_map = {label: i for i, label in enumerate(np.unique(labels))}
                labels = torch.LongTensor([label_map[label.item()] for label in labels])
                sens = torch.Tensor(sens[label_idx])

                data = Data(edge_index=edge_index, num_nodes=features.shape[0], x=features, y=labels, sens=sens)

                # get largest connected component of data
                data = T.LargestConnectedComponents()(data)

                edge_index = data.edge_index
                features = data.x.to(device)
                labels = data.y
                sens = data.sens

            torch.save({'features': features, 'labels': labels, 'sens': sens, 'edge_index': edge_index}, f'real_datasets/facebook/{dataset}_processed.pt')

        elif 'tolokers' in dataset:
            data = np.load('real_datasets/tolokers/tolokers.npz')
            features = torch.Tensor(data['node_features']).to(device)

            # the education level (self-reported) is one-hot encoded alreayd
            labels = data['node_labels']
            # sens will be if they passed english test -- justify this 
            sens = features[:, -1]

            edge_index = data['edges'].T

            torch.save({'features': features, 'labels': labels, 'sens': sens, 'edge_index': edge_index}, f'real_datasets/tolokers/tolokers_processed.pt')
            # label indx helps us handle the semi-sup case where not all data has labels 

    else:
        #load from memory the subgraphs/transformed data 
        if 'fb100' in dataset:
            orig_dataset = dataset 
            if dataset == 'fb100-penn94':
                dataset = 'Penn94'
            path = 'facebook'
        else:
            orig_dataset = dataset 
            path = dataset 
        
        data = torch.load(f'real_datasets/{path}/{dataset}_processed.pt')
        labels = data['labels']
        features = data['features']
        sens = data['sens']
        if a and b:
            edge_index = torch.load('synthetic_datasets/synth_edge/{4}_synth_edge_index_weight_degree_{0}_a_{1}_b_{2}_num_bins_{3}.pt'.format(False, a, b, 5, orig_dataset))
        else: 
            edge_index = data['edge_index']

    labels = torch.LongTensor(labels).to(device)
    sens = torch.Tensor(sens).to(device)
    edge_index = to_undirected(torch.LongTensor(edge_index).to(device))

    if stratify:
        # Use power and seed from arguments 
        if a and b: 
            file_name = f'ood_splits/{dataset}_{a}_{b}_splits_class_{class_power}_{seed}.npz'
        else: 
            file_name = f'ood_splits/{dataset}_splits_class_{class_power}_{seed}.npz'
        if not os.path.exists(file_name):
            print('Stratifying Data with Power {0}, Seed {1}, A {2}, B {3}'.format(class_power, seed, a, b))
            gen_splits(labels, sens, edge_index, dataset, class_power, seed, plot=True, file_name=file_name)
        else:
            print('Already processed')
        split_indices = np.load(file_name)
        idx_train = split_indices['idx_train']
        idx_val = split_indices['idx_val']
        idx_test = split_indices['idx_test']
    else: 
        label_idx = torch.arange(len(labels))
        idx_train = label_idx[:int(0.5 * len(label_idx))]
        idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    return features, labels, sens, edge_index, idx_train, idx_val, idx_test

def gen_splits(labels, sens, edge_index, dataset, class_power, seed, plot=True, num_bins=5, file_name=None): 
    
    # get local homophily ratio
    local_homophily_labels, num_neighs = get_local_homophily(edge_index, labels, radius=1)

    # label homophily processing
    local_homophily_labels[np.argwhere(np.isnan(local_homophily_labels))] = 0
    label_hist, _ = np.histogram(local_homophily_labels, bins=num_bins)
    distr = (label_hist**(class_power))/np.sum(label_hist**(class_power))
    
    # rows are labels
    # cols are sens
    inv_distr = (distr**(-1))/np.sum(distr**(-1))

    # stratify based on local homophily and global homophily
    idx_train, idx_val, idx_test = stratify_idx(local_homophily_labels, distr, inv_distr, seed)  
    
    # save the splits
    np.savez(file_name, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == '__main__':

    # plot subplots for each dataset showing sorted distribution of unique labels and sens
    seeds = [123, 246, 492]
    shape_params = [None, (3.0, 10.0), (3.0,5.0), (5.0,3.0), (10.0,3.0)]
    datasets = ['pokec', 'fb100-penn94', 'tolokers']
    gammas = [3.0]

    for dataset in datasets:
        for seed in seeds: 
            for shape_param in shape_params:     
                for class_power in gammas:
                    if shape_param is None:
                        a = None
                        b = None
                    else: 
                        a = shape_param[0]
                        b = shape_param[1]
                    features, labels, sens, edge_index, idx_train, idx_val, idx_test = load_dataset(dataset, 'cpu', stratify=True, labeled_subgraph=True, \
                                                                                                    in_mem=True, seed=seed, class_power=class_power, a=a, b=b)
                    


        




        
        







