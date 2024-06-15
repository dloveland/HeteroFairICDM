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
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import convert, is_undirected, to_undirected, homophily, subgraph
from scipy.stats import beta, wasserstein_distance
import ot
import tqdm 
import copy
import argparse 
import matplotlib

def load_graph(dataset, in_mem=True):

    #load from memory the subgraphs/transformed data 
    if 'fb100' in dataset:
        if dataset == 'fb100-cornell5':
            dataset = 'Cornell5'
        elif dataset == 'fb100-johns_hopkins55':
            dataset = 'Johns Hopkins55'
        elif dataset == 'fb100-penn94':
            dataset = 'Penn94'
        path = 'facebook'
    else:
        path = dataset 
    
    data = torch.load(f'../real_datasets/{path}/{dataset}_processed.pt')
    labels = data['labels']
    features = data['features']
    sens = data['sens']
    edge_index = data['edge_index']

    device = 'cpu'
    labels = torch.LongTensor(labels).to(device)
    sens = torch.Tensor(sens).to(device)
    edge_index = to_undirected(torch.LongTensor(edge_index).to(device))

    return features, labels, sens, edge_index

def get_local_homophily(edge_index, metric, radius=1):

    all_local_homophily = []
    num_neighs = []

    num_nodes = len(metric)

    for node in range(num_nodes):

        neighbors = edge_index[0][edge_index[1] == node]
    
        neighbor_metrics = metric[neighbors]

        num_neighs.append(len(neighbors))
        same = (neighbor_metrics == metric[node]).sum()

        local_homophily = same / float(len(neighbor_metrics))
  
        all_local_homophily.append(local_homophily)
        
    return np.array(all_local_homophily), np.array(num_neighs)

def get_global_homophily(edge_index, metric):

    global_homophily = homophily(edge_index, metric, method='edge')
    return global_homophily

def get_beta(a, b, num_nodes):

    r = beta.rvs(a, b, size=num_nodes)
    return r

def get_hist(val, num_bins):

    # get binned values of r normalized
    hist, bin_edges = np.histogram(val, bins=num_bins, range=(0,1))
    hist = hist / float(hist.sum())

    return hist, bin_edges


def emd(X, Y, num_bins=5):

    bin_vals = np.linspace(0, 1, num_bins+1)[:-1]
    # Setup grid
    A, B = np.meshgrid(bin_vals, bin_vals)
    print(A, B)
    diff = np.sqrt((np.subtract(A, B)**2))
    # Compute the distance matrix
    print(diff)
    print(X.shape, Y.shape, diff.shape)
    wass = ot.emd2(X.flatten(), Y.flatten(), diff)

    return wass  


    

if __name__ == '__main__':

    # set up param pairs from the sbatch runs
    #pairs = [(10.0, 3.0), (3.0, 10.0), (20.0, 50.0), (5.0, 2.0), (25.0, 10.0), (50.0, 20.0), (3.0, 3.0), (5.0, 5.0), (3.0, 2.0), (3.0, 5.0), (2.0, 3.0), (2.0, 5.0), (5.0, 3.0), (10.0, 25.0)]
    pairs = [(3.0, 10.0), (3.0,5.0), (5.0,3.0), (10.0,3.0)]
    

    results = []
    xs = []
    ys = []
    # set seeds
    np.random.seed(123)
    random.seed(123)

    font = {'size'   : 15}

    matplotlib.rc('font', **font)

    for pair in pairs:
        a = pair[0]
        b = pair[1]
        print(a, b)
        num_bins = 5

        weight_degree = False
        dataset = 'tolokers'
        dataset_name = 'Tolokers'
        #dataset = 'pokec'
        #dataset_name = 'Pokec'
        #dataset = 'fb100-cornell5'
        #dataset_name = 'FB-Cornell5'
        #dataset = 'fb100-penn94'
        #dataset_name = 'FB-Penn94'
        features, labels, sens, edge_index = load_graph(dataset)
        print(get_global_homophily(edge_index, labels))
        print(get_global_homophily(edge_index, sens))
        
        local_homophily, _ = get_local_homophily(edge_index, labels)

        base_hist, base_bins = get_hist(local_homophily, 10)

        plt.figure()
        
        plt.title(dataset_name)
        plt.bar((np.arange(10)/10)+0.05, base_hist, alpha=0.5, width=1/10, edgecolor='black', linewidth=1.4)
        plt.ylabel("Frequency")
        plt.xlabel('Local Homophily Levels')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.savefig(f'paper_figs/{dataset}', bbox_inches='tight')
        sys.exit()

        # set up target dist
        beta_dist = get_beta(a, b, 1000)
        xs.append(beta_dist.mean())
        ys.append(beta_dist.std())
        target_hist, target_bins = get_hist(beta_dist, num_bins)

        # set up orig dist
        
        # load in edge index from file
        try:
            modified_edge_index = torch.load('synth_edge/{4}_synth_edge_index_weight_degree_{0}_a_{1}_b_{2}_num_bins_{3}.pt'.format(weight_degree, a, b, num_bins, dataset))
        except:
            continue 
        # get local homophily of modified edge index
        local_homophily, _ = get_local_homophily(modified_edge_index, labels)
        modified_hist, modified_bins = get_hist(local_homophily, num_bins)

        # compute the area between the two histograms
        orig_area = np.round(emd(base_hist, target_hist), 3)
        print('Orig Distance: ', orig_area)
        modified_area = np.round(emd(modified_hist, target_hist), 3)
        print('Modified Distance: ', modified_area)

        results.append((orig_area, modified_area))
        
        color_one = '#762A83'
        color_three = '#e67386'
        color_two = '#1B7837'
        # plot subplots comparing target hist to base hist and modified hist
        plt.subplots(1, 2, sharex=True, sharey=True)
        plt.subplot(1, 2, 1)
        plt.bar(base_bins[:-1]+((1/num_bins)/2), base_hist, color=color_three, alpha=0.5, width=1/num_bins, label='Original', edgecolor='black', linewidth=1.2)
        plt.bar(target_bins[:-1]+((1/num_bins)/2), target_hist, color=color_two, alpha=0.5, width=1/num_bins, label='Goal', edgecolor='black', linewidth=1.2, hatch="/")
        # add text for orig_area
        plt.text(0.5, 0.7, 'EMD: {0}'.format(orig_area), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes, weight="bold")
        plt.title('Original Distr.')
        plt.xlabel('Local Homophily')
        plt.ylabel('Frequency')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.bar(modified_bins[:-1]+((1/num_bins)/2), modified_hist, color=color_one, alpha=0.5, width=1/num_bins, label='Modified', edgecolor='black', linewidth=1.2)
        plt.bar(target_bins[:-1]+((1/num_bins)/2), target_hist, color=color_two, alpha=0.5, width=1/num_bins, label='Goal', edgecolor='black', linewidth=1.2, hatch="/")
        plt.text(0.5, 0.7, 'EMD: {0}'.format(modified_area), fontsize=12, ha='center', va='center', transform=plt.gca().transAxes, weight="bold")
        plt.title('Modified Distr.')
        plt.xlabel('Local Homophily')
        plt.tight_layout()
        plt.legend()
        #plt.suptitle('a: {0}, b: {1}, E[h]: {2}'.format(a, b, a/(float(a)+b)), weight='bold', y=1.01)
        plt.suptitle('a: {0}, b: {1}'.format(a, b, a/(float(a)+b)), weight='bold', y=1.01)
        plt.savefig('paper_figs/{0}_goal_vs_actual_num_bins_{1}_dist_a_{2}_b_{3}.png'.format(dataset, num_bins, a, b), bbox_inches='tight')
        

    # 3d scatter plot where x is a, y is b, z is emd
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #xs = [pair[0] for pair in pairs]
    #ys = [pair[1] for pair in pairs]
    zs = [result[1] for result in results]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('Mean of Beta Distr')
    ax.set_ylabel('Std of Beta Distr')
    ax.set_zlabel('EMD')
    plt.title('EMD between Goal and Modified Distribution')
    plt.show()


