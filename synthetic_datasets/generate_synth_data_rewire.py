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
import math
import argparse 

def emd(X, Y, num_bins=5):

    bin_vals = np.linspace(0, 1, num_bins+1)[:-1]
    # Setup grid
    A, B = np.meshgrid(bin_vals, bin_vals)
    diff = np.sqrt((np.subtract(A, B)**2))
    # Compute the distance matrix
    wass = ot.emd2(X.flatten(), Y.flatten(), diff)

    return wass  

def load_graph(dataset, in_mem=True):

    #load from memory the subgraphs/transformed data 
    if 'fb100' in dataset:
        if dataset == 'fb100-cornell5':
            dataset = 'Cornell5'
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

def get_beta(a, b, num_nodes):

    r = beta.rvs(a, b, size=num_nodes)
    return r

def get_hist(val, num_bins):

    # get binned values of r normalized
    hist, bin_edges = np.histogram(val, bins=num_bins, range=(0,1))
    hist = hist / float(hist.sum())

    return hist, bin_edges

def plot_opt_transport(T, base_bins, base_hist, target_hist, dataset, title):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    ax.imshow(T, cmap='Blues')
    for (j,i),label in np.ndenumerate(T):
        ax.text(i,j,round(label, 3),ha='center',va='center', fontsize=10)
    ax_histy.barh((base_bins[:-1])*(len(base_bins)-1), base_hist, color='red', alpha=0.8)
    ax_histx.bar((base_bins[:-1])*(len(base_bins)-1), target_hist, color='purple', alpha=0.8)#, color='green', orientation='horizontal', alpha=0.5)
    
    #ax.set_xticks(np.arange(0, 6, 1))
    #ax.set_yticks(np.arange(0, 6, 1))
    ax_histx.tick_params(axis="x", labelbottom=False, labelleft=True, labeltop=False, labelright=False, \
                         top=False, bottom=False, left=False, right=False)
    ax_histy.tick_params(axis="y", labelleft=False, labelbottom=False, labeltop=False, labelright=False)
    #ax_histx.set_yticklabels([])
    #ax_histy.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Goal Local Homophily Distribution', fontsize=13)
    ax.set_ylabel('Actual Local Homophily Distribution', fontsize=13)
    #ax.legend()
    plt.suptitle(title, weight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/{2}_goal_vs_actual_{0}_num_bins_{1}.png'.format(title, len(base_bins)-1, dataset))


def smooth_weight(drift, direction, temp=1):

    # if we need to increase drift, priotiize negatives
    if direction == 'invert':
        drift = -1 * drift
    
    # otherwise we just use drift as is priotiizing positives since we want to decrease drift
    #sofmax the drift vector
    drift = np.exp(drift / temp) / np.sum(np.exp(drift/temp))

    return drift


def alter_local_homophily(edge_index, metr, a, b, num_bins=5, weight_degree=False, dataset=None):
    
    # set seeds
    np.random.seed(123)
    random.seed(123)

    metric = metr.clone()
    num_nodes = len(metric)
    
    # set up target dist
    beta_dist = get_beta(a, b, num_nodes)
    
    target_hist, target_bins = get_hist(beta_dist, num_bins)
 
    # set up base dist
    local_homophily, _ = get_local_homophily(edge_index, metric)
    base_hist, base_bins = get_hist(local_homophily, num_bins)
  
    T = ot.emd_1d(base_bins[:-1], base_bins[:-1], a=base_hist, b=target_hist)
    row_sums = T.sum(axis=1)
    T = np.divide(T, row_sums[:, np.newaxis], out=np.zeros_like(T), where=row_sums!=0)

    # compute the area between the two histograms
    orig_dist = np.round(emd(base_hist, target_hist), 3)

    

    T_scaled = np.zeros(T.shape)
    #plot_opt_transport(T, base_bins, base_hist, target_hist)
    # Preprocess T so that we know how many nodes to move from each bin to each other bin
    for base_b in range(num_bins):
        bin_idx = np.where((local_homophily >= base_bins[base_b]) & (local_homophily < base_bins[base_b+1]))[0]
        # go through each target bin 
        for tar_b in range(num_bins):
            # if we are looking at same base and target bin, dont need to do anything since nothing to move
            if base_b == tar_b:
                continue
            amount_to_move = int(math.floor(T[base_b, tar_b] * len(bin_idx)))
            T_scaled[base_b, tar_b] = amount_to_move
    
    plot_opt_transport(T, base_bins, base_hist, target_hist, dataset, "Initial Distribution | {0}, {1}, {2}".format(weight_degree, a, b))
 
    node_score = {}

    for node in range(num_nodes):
        node_score[node] = 0

    node_labels = {}
    node_degree = {}
    node_same = {}
    labels_to_node = {}
    init_local_homophily = {}
    target_local_homophily = {}
   
    for val in metric.unique():
        labels_to_node[val.item()] = np.where(metric == val)[0]

    # go through all bins 
    print('Find set of nodes to modify: ')
    for base_b in tqdm.tqdm(range(num_bins)):
        bin_idx = np.where((local_homophily >= base_bins[base_b]) & (local_homophily < base_bins[base_b+1]))[0]
        # shuffle bin_idx
        random.shuffle(bin_idx)
        
        # go through each target bin 
        for tar_b in range(num_bins):
            mid_bin = np.round(((tar_b + 1) * (1/float(num_bins)) + (tar_b) * (1/float(num_bins)))/2, 2)
            
            # get number of data points needed to be moved
            amount_to_move = int(T_scaled[base_b, tar_b])
            for move_idx in range(amount_to_move):
               
                # get the node to move
                node = bin_idx[move_idx]
                
                # get degree of node
                degree = (len(edge_index[1][edge_index[0] == node]))
                homophily_level = local_homophily[node]

                num_neighs_same = int(np.round(homophily_level*degree))
                # adding homophilous 
                necessary_change = mid_bin*degree - num_neighs_same
                
                if abs(necessary_change) >= 1:
                    necessary_change = int(np.sign(necessary_change) * np.ceil(np.abs(necessary_change)))
                    node_score[node] = necessary_change
                    node_labels[node] = metric[node].item()
                    node_degree[node] = degree
                    node_same[node] = num_neighs_same
                    target_local_homophily[node] = mid_bin
                    init_local_homophily[node] = homophily_level
                    #print(mid_bin, local_homophily[node], mid_bin-local_homophily[node], necessary_change, degree)
                
                
            bin_idx = bin_idx[amount_to_move:]

    print('Perform Re-wiring: ')
    nodes_to_move = np.array(list(node_score.keys()))
    node_orig_degree = copy.deepcopy(node_degree)
    
    # REWIRING PHASE 
    # iterate over nodes to rewire
    num_edits = 0
    for node in tqdm.tqdm(nodes_to_move):
        num_moves = node_score[node]
        # this node may have already been solved, we can just continue 
        if num_moves == 0:
            continue
        label = node_labels[node]
 
        # get neighbors of node and add node to neighbors 
        neighbors = (edge_index[1][edge_index[0] == node]).numpy()
        
        # if move_score is positive, we need to add homophilous edges or remove heterophilous edges 
        if num_moves > 0:
            # check scores of heterophilous neighbors
            # in re-wiring we will find heterophilous edges with positive score 
            try:
                heterophilous_neighbors = neighbors[np.array(labels[neighbors] != label)]
            except Exception as e:
                sys.exit()
            heterophilous_move_score = np.array([node_score[neigh_node] for neigh_node in heterophilous_neighbors])
            pos_heterophilous = np.where(heterophilous_move_score > 0)[0]
            # sample up to num_moves from pos_homophilous
            if len(pos_heterophilous) < num_moves:
                num_moves = len(pos_heterophilous)
            
            # we will delete the edges between node and deletions 
            deletions = np.random.choice(heterophilous_neighbors[pos_heterophilous], num_moves, replace=False)
           
            # remove edges from edge_index
            for neigh_del in deletions:
                # delete both directions 
                edge_index = edge_index[:, ~((edge_index[0] == node) & (edge_index[1] == neigh_del))]
                edge_index = edge_index[:, ~((edge_index[0] == neigh_del) & (edge_index[1] == node))]
                # update degree value for node and remove one heterophilous edge 
                node_degree[node] -= 1
                node_degree[neigh_del] -= 1
                # update neighs number of connections with change in degree and node same 
                necessary_change = target_local_homophily[neigh_del] * (node_degree[neigh_del]) - (node_same[neigh_del])
                # update node score for neigh del 
                node_score[neigh_del] = int(np.sign(necessary_change) * np.ceil(np.abs(necessary_change)))
                
            # get all nodes from node_labels that have the same label as node
            same_label = labels_to_node[label]

            # get all nodes that have a positive move score
            move_score_greater_zero = np.array([s for s in node_score if node_score[s] > 0])
         
            # take intersection between the two as those with pos idx and those that have move_score greater than 0
            poss_additions = nodes_to_move[np.intersect1d(same_label, move_score_greater_zero)]
            # remove neighbors and node from poss_additions
            poss_additions = np.setdiff1d(poss_additions, neighbors)
            poss_additions = np.setdiff1d(poss_additions, node)
            if len(poss_additions) < num_moves:
                num_moves = len(poss_additions)
            # randomly sample from poss_additions to make the additions
            additions = np.random.choice(poss_additions, num_moves, replace=False)

            # add edges to edge_index
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor([node]*num_moves), torch.LongTensor(additions)])], dim=1)
            # add the opposite direction edge
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor(additions), torch.LongTensor([node]*num_moves)])], dim=1)

            # if any additions are an index higher then len(metrics), break
            if np.any(additions >= len(metric)):
                sys.exit()

            # change move_score for the nodes that were added
            for add in additions:
                node_degree[node] += 1
                node_degree[add] += 1
                node_same[node] += 1
                node_same[add] += 1
                
                # update node score of add after the change 
                necessary_change = target_local_homophily[add] * (node_degree[add]) - (node_same[add])
                node_score[add] = int(np.sign(necessary_change) * np.ceil(np.abs(necessary_change)))

            # finally process and update node's value
            node_score[node] = target_local_homophily[node] * (node_degree[node]) - (node_same[node])


        else:
            # if move_score is negative, we need to add heterophilous edges or remove homophilous edges 
            # check scores of homophilous neighbors
            # in re-wiring we will find homophilous edges with a negative score as this connection wants more heterophilous edges 
            homophilous_neighbors = neighbors[np.array(labels[neighbors] == label)]
            homophilous_move_score = np.array([node_score[neigh_node] for neigh_node in homophilous_neighbors])
            neg_homophilous = np.where(homophilous_move_score < 0)[0]

            # sample up to num_moves from pos_homophilous
            if len(neg_homophilous) < abs(num_moves):
                num_moves = len(neg_homophilous)
            else:
                num_moves = abs(num_moves)
            
            # we will delete the edges between node and additions
            deletions = np.random.choice(homophilous_neighbors[neg_homophilous], num_moves, replace=False)

            # remove edges from edge_index
            for neigh_del in deletions:
                # remove edges
                edge_index = edge_index[:, ~((edge_index[0] == node) & (edge_index[1] == neigh_del))]
                edge_index = edge_index[:, ~((edge_index[0] == neigh_del) & (edge_index[1] == node))]

                # update degree value for node and remove one homopphilous edge 
                node_degree[node] -= 1
                node_degree[neigh_del] -= 1
                node_same[node] -= 1
                node_same[neigh_del] -= 1

                ## update neighbor as we just changed degree 
                necessary_change = target_local_homophily[neigh_del] * (node_degree[neigh_del]) - (node_same[neigh_del])
                node_score[neigh_del] = int(np.sign(necessary_change) * np.ceil(np.abs(necessary_change)))
           

            # get the node labels that differ from this, and then concat all their nodes 
            all_labels = list(labels_to_node.keys())
            all_labels.remove(label)
            
            diff_label = []
            for label_key in all_labels:
                diff_label.extend(labels_to_node[label_key])
            
            #diff_label = labels_to_node[abs(label - 1)] 
            # get all nodes that have a negative move score
            move_score_lesser_zero = np.array([s for s in node_score if node_score[s] < 0])
        
            # take intersection between the two as those with diff label and those that have move_score less than zero
            poss_additions = nodes_to_move[np.intersect1d(diff_label, move_score_lesser_zero)]
       
            # remove neighbors and node from poss_additions
            poss_additions = np.setdiff1d(poss_additions, neighbors)
            poss_additions = np.setdiff1d(poss_additions, node)
            # randomly sample from poss_additions to make the additions
            if len(poss_additions) < num_moves:
                num_moves = len(poss_additions)

            additions = np.random.choice(poss_additions, num_moves, replace=False)

            # add edges to edge_index
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor([node]*num_moves), torch.LongTensor(additions)])], dim=1)
            # add the opposite direction edge
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor(additions), torch.LongTensor([node]*num_moves)])], dim=1)

            # if any additions are an index higher then len(metrics), break
            if np.any(additions >= len(metric)):
                sys.exit()

            # change move_score for the nodes that were added
            for add in additions:
                node_degree[node] += 1
                node_degree[add] += 1

                # update node score of add after the change 
                necessary_change = target_local_homophily[add] * (node_degree[add]) - (node_same[add])
                node_score[add] = int(np.sign(necessary_change) * np.ceil(np.abs(necessary_change)))

            # finally process and update node's value
            node_score[node] = target_local_homophily[node] * (node_degree[node]) - (node_same[node])

    # Now we will only perform additions, thus we will recompute node_score using the only additions equation
    print('Readjust Node Scores for Additions: ')
    for node in tqdm.tqdm(nodes_to_move):
        if node_score[node] == 0:
            continue 
        degree = node_degree[node]
        same_count = node_same[node]
        diff_count = degree - same_count
        label = node_labels[node]
        target_homophily = target_local_homophily[node]
        
        score_dir = np.sign(node_score[node])

        curr_homophily = float(same_count)/float(degree)

        # need more homophilous edges
        if score_dir > 0:
            # if curr homophily is greater than target homophily, we are done 
            if curr_homophily >= target_homophily:
                node_score[node] = 0
                continue
            else:
                node_score[node] = int(np.ceil(((target_homophily * degree) - same_count) / (1-target_homophily)))
        elif score_dir < 0:
            # if curr homophily is less than target homophily, we are done 
            if curr_homophily <= target_homophily:
                node_score[node] = 0
                continue
            else: 
                node_score[node] = int(-1*np.ceil((((1 - target_homophily) * degree) - diff_count) / target_homophily))
    
    # update the local homophily variable with updated metric 
    local_homophily, _ = get_local_homophily(edge_index, metric)

    result_hist, result_bins = get_hist(local_homophily, num_bins)

    T = ot.emd_1d(result_bins[:-1], result_bins[:-1], a=result_hist, b=target_hist)
    row_sums = T.sum(axis=1)
    T = np.divide(T, row_sums[:, np.newaxis], out=np.zeros_like(T), where=row_sums[:, np.newaxis]!=0.0)

    # update T 
    T_scaled = np.zeros(T.shape)
    #plot_opt_transport(T, base_bins, base_hist, target_hist)
    # Preprocess T so that we know how many nodes to move from each bin to each other bin
    for base_b in range(num_bins):
        bin_idx = np.where((local_homophily >= base_bins[base_b]) & (local_homophily < base_bins[base_b+1]))[0]
        # go through each target bin 
        for tar_b in range(num_bins):
            # if we are looking at same base and target bin, dont need to do anything since nothing to move
            if base_b == tar_b:
                continue
            amount_to_move = round(T[base_b, tar_b] * len(bin_idx))
            T_scaled[base_b, tar_b] = amount_to_move
    #print(T_scaled)

    plot_opt_transport(T, result_bins, result_hist, target_hist, dataset, 'Rewiring Only Phase | {0}, {1}, {2}'.format(weight_degree, a, b))

    rewire_dist = np.round(emd(result_hist, target_hist), 3)


    print('Perform Additions to Finalize Distribution Change: ')
    for node in tqdm.tqdm(nodes_to_move):
        num_moves = node_score[node]
        if num_moves == 0:
            continue
        label = node_labels[node]

        # this node may have already been solved, we can just continue
        # otherwise we can add edges 
        if node_score[node] > 0:
            # need to add homophilous edges to turn more homophilous 
            # get all nodes from node_labels that have the same label as node
            same_label = labels_to_node[label]

            # get all nodes that have a positive move score
            move_score_greater_zero = np.array([s for s in node_score if node_score[s] > 0])
         
            # take intersection between the two as those with pos idx and those that have move_score greater than 0
            poss_additions = nodes_to_move[np.intersect1d(same_label, move_score_greater_zero)]
            # remove neighbors and node from poss_additions
            poss_additions = np.setdiff1d(poss_additions, neighbors)
            poss_additions = np.setdiff1d(poss_additions, node)
            # no possible additions to be made 
            if len(poss_additions) == 0:
                continue
            if len(poss_additions) < num_moves:
                num_moves = len(poss_additions)
            # randomly sample from poss_additions to make the additions
            if weight_degree:
                # if we weight degree, we should prioritize nodes with degrees that have changed more substantially during the optimization process
                # for each poss_addition, we will calculate the change in degree and use that as the weight
                weights = np.array([abs(node_degree[add] - node_orig_degree[add]) for add in poss_additions])
                weights = smooth_weight(weights, 'invert', temp=1)
                # if nan in weights
                if np.isnan(weights).any():
                    # use a uniform distribution
                    weights = np.ones(len(poss_additions)) / len(poss_additions)
                additions = np.random.choice(poss_additions, num_moves, replace=False, p=weights)
            else:       
                additions = np.random.choice(poss_additions, num_moves, replace=False)

            # if any additions are an index higher then len(metrics), break
            if np.any(additions >= len(metric)):
                print([node]*num_moves)
                print(additions)
                print(len(metric))
                sys.exit()

            # add edges to edge_index
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor([node]*num_moves), torch.LongTensor(additions)])], dim=1)
            # add the opposite direction edge
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor(additions), torch.LongTensor([node]*num_moves)])], dim=1)

            # change move_score for the nodes that were added, added num_moves worth of homophilous edges 
            for add in additions:
                node_score[add] -= 1

            # finally process and update node's value
            node_score[node] -= num_moves
        elif node_score[node] < 0:
            # get all nodes that have a differnt label as node
            all_labels = list(labels_to_node.keys())
            all_labels.remove(label)
            
            diff_label = []
            for label_key in all_labels:
                diff_label.extend(labels_to_node[label_key])

            # get all nodes that have a negative move score
            move_score_lesser_zero = np.array([s for s in node_score if node_score[s] < 0])
        
            # take intersection between the two as those with diff label and those that have move_score less than zero
            poss_additions = nodes_to_move[np.intersect1d(diff_label, move_score_lesser_zero)]
       
            # remove neighbors and node from poss_additions
            poss_additions = np.setdiff1d(poss_additions, neighbors)
            poss_additions = np.setdiff1d(poss_additions, node)
            # randomly sample from poss_additions to make the additions
            num_moves = abs(num_moves)
            if len(poss_additions) == 0:
                continue

            if len(poss_additions) < num_moves:
                num_moves = len(poss_additions)
            
            if weight_degree:
                # if we weight degree, we should prioritize nodes with degrees that have changed more substantially during the optimization process
                # for each poss_addition, we will calculate the change in degree and use that as the weight
                weights = np.array([abs(node_degree[add] - node_orig_degree[add]) for add in poss_additions])
                weights = smooth_weight(weights, 'invert', temp=1)
                print(weights)
                # if nan in weights
                if np.isnan(weights).any():
                    # use a uniform distribution
                    weights = np.ones(len(poss_additions)) / len(poss_additions)
                additions = np.random.choice(poss_additions, num_moves, replace=False, p=weights)
            else:       
                additions = np.random.choice(poss_additions, num_moves, replace=False)

            # add edges to edge_index
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor([node]*num_moves), torch.LongTensor(additions)])], dim=1)
            # add the opposite direction edge
            edge_index = torch.cat([edge_index, torch.stack([torch.LongTensor(additions), torch.LongTensor([node]*num_moves)])], dim=1)

             # if any additions are an index higher then len(metrics), break
            if np.any(additions >= len(metric)):
                print([node]*num_moves)
                print(additions)
                print(len(metric))
                sys.exit()
                

            # change move_score for the nodes that were added
            for add in additions:
                node_score[add] += 1

            # finally process and update node's value
            node_score[node] += num_moves
  
            
    # update the local homophily variable with updated metric 
    local_homophily, _ = get_local_homophily(edge_index, metric)

    result_hist, result_bins = get_hist(local_homophily, num_bins)

    T = ot.emd_1d(result_bins[:-1], result_bins[:-1], a=result_hist, b=target_hist)
    row_sums = T.sum(axis=1)
    T = np.divide(T, row_sums[:, np.newaxis], out=np.zeros_like(T), where=row_sums[:, np.newaxis]!=0.0)

    # update T 
    T_scaled = np.zeros(T.shape)
    #plot_opt_transport(T, base_bins, base_hist, target_hist)
    # Preprocess T so that we know how many nodes to move from each bin to each other bin
    for base_b in range(num_bins):
        bin_idx = np.where((local_homophily >= base_bins[base_b]) & (local_homophily < base_bins[base_b+1]))[0]
        # go through each target bin 
        for tar_b in range(num_bins):
            # if we are looking at same base and target bin, dont need to do anything since nothing to move
            if base_b == tar_b:
                continue
            amount_to_move = round(T[base_b, tar_b] * len(bin_idx))
            T_scaled[base_b, tar_b] = amount_to_move
    #print(T_scaled)

    rewire_add_dist = np.round(emd(result_hist, target_hist), 3)

    orig_dist, rewire_dist, rewire_add_dist 

    plot_opt_transport(T, result_bins, result_hist, target_hist, dataset, 'Rewiring and Additions Phase | {0}, {1}, {2}'.format(weight_degree, a, b))


    return edge_index, orig_dist, rewire_dist, rewire_add_dist 

if __name__ == '__main__':

    # use argparse to pass in two arguments, a and b
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=3.0)
    parser.add_argument('--b', type=float, default=10.0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_bins', type=int, default=5)
    args = parser.parse_args()
    a = args.a
    b = args.b
    num_bins = args.num_bins

    weight_degree = False
    dataset = args.dataset
    features, labels, sens, edge_index = load_graph(dataset)
    #features, labels, sens, edge_index, label_idx = load_graph('bail', path='../real_datasets/bail')
    print(get_global_homophily(edge_index, labels))
    print(get_global_homophily(edge_index, sens))

    # get degree distribution
    orig_degrees = np.zeros(len(features))
    for node in range(len(features)):
        orig_degrees[node] = len(edge_index[1][edge_index[0] == node])
    # bin degrees
    orig_hist, orig_bins = np.histogram(orig_degrees, 10)

    # get new edge index 
    edge_index, orig_dist, rewire_dist, rewire_add_dist  = alter_local_homophily(edge_index, labels, a, b, num_bins=num_bins, weight_degree=weight_degree, dataset=dataset)

    # get new degree distribution
    new_degrees = np.zeros(len(features))
    for node in range(len(features)):
        new_degrees[node] = len(edge_index[1][edge_index[0] == node])
    # bin degrees
    new_hist, new_bins = np.histogram(new_degrees, 10)

    print(orig_hist)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(orig_bins[:-1], orig_hist, alpha=0.5, label='Original')
    ax.plot(new_bins[:-1], new_hist, alpha=0.5, label='Modified')

    ax.set_title('Degree Distribution')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('figures/{4}_degree_dist_weight_degree_{0}_a_{1}_b_{2}_bins_{3}.png'.format(weight_degree, a, b, num_bins, dataset))

    # save the new edge index using a and b in title
    torch.save(edge_index, 'synth_edge/{4}_synth_edge_index_weight_degree_{2}_a_{0}_b_{1}_num_bins_{3}.pt'.format(a, b, weight_degree, num_bins, dataset))
    
    np.save(f'edit_metrics/{dataset}_{a}_{b}', np.array([orig_dist, rewire_dist, rewire_add_dist]))
    
