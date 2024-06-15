import os
import torch
from models import *
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

def KL(P,Q):

    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence

def JS(P, Q):

    M = 0.5 * (P + Q)
    return 0.5 * KL(P, M) + 0.5 * KL(Q, M)

# Bhattacharyya Distance
def BD(P, Q):
    return -np.log(np.sum(np.sqrt(P*Q)))


def emd(X, Y, num_bins=5):

    bin_vals = np.linspace(0, 1, num_bins)
    # Setup grid
    A, B = np.meshgrid(bin_vals, bin_vals)
    coords = np.array([A.flatten(), B.flatten()]).T
    
    expand_c_a = coords[:, None, :]
    expand_c_b = coords[None, :, :]
    diff = np.sqrt(np.sum(np.subtract(expand_c_a, expand_c_b)**2, axis=2))
    
    wass = ot.emd2(X.flatten(), Y.flatten(), diff)

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
            model = LINKX(num_nodes=num_nodes, in_channels=num_feats, hidden_channels=hidden, num_layers=depth, out_channels=self.nclass, dropout=dropout)
        else:
            raise ValueError('Model type not supported')

        model.load_state_dict(torch.load(self.weights_path))
        self.model = model 

        perf, ax_perf = plt.subplots()
        parity, ax_parity = plt.subplots()

    def eval_model(self, features, labels, eval_idx, edge_index):

        self.model.eval()
        output = self.model(features, edge_index)
        if self.nclass == 1:
            # Binary Cross-Entropy
            preds = (output.squeeze()>0).type_as(labels)
        else:
            # Cross-Entropy
            preds = torch.argmax(output, axis=1)
        preds = preds[eval_idx].cpu().numpy()
        f1 = f1_score(labels[eval_idx].cpu().numpy(), preds, average='binary' if self.nclass == 1 else 'micro')
        #rocauc = roc_auc_score(labels[eval_idx].cpu().numpy(), preds)
        return f1, preds

    def eval_1D_strat(self, preds, preds_mlp, labels, eval_idx, edge_index, sens, stratify_width=1.0, dataset=None):

        #new_edge_index = to_undirected(edge_index)
        data = Data(edge_index=edge_index, y=labels, sens=sens, num_nodes=len(labels))

        global_class = homophily(edge_index, labels, method='edge')
        global_sens = homophily(edge_index, sens, method='edge')
   
        # print global homophily for class and sens
        #print("Global Homophily Class: {}".format(global_homophily_class))
        #print("Global Homophily Sens: {}".format(global_homophily_sens))
        
        local_homophily_class = get_local_homophily(edge_index, labels)
        local_homophily_sens = get_local_homophily(edge_index, sens)

        test_local_homophily_class = local_homophily_class[eval_idx]
        test_local_homophily_sens = local_homophily_sens[eval_idx]

        test_labels = labels[eval_idx]
        test_sens = sens[eval_idx]
        
        f1s = []
        f1s_mlp = []
        parities = []
        parities_mlp = []

        # get data points with local homophily between bins of 0.1 width
        stratify_width = 0.2
        num_bins = int(1/stratify_width)
        bins = np.linspace(0, 0.8, num=num_bins)
        for bin_start in bins:
            bin_end = bin_start + stratify_width
            bin_idx = np.where((test_local_homophily_class >= bin_start) & (test_local_homophily_class < bin_end))[0]
            bin_labels = test_labels[bin_idx]
            bin_sens = test_sens[bin_idx]

            #print(len(bin_idx))
            if len(bin_idx) < 3:
                continue 
            bin_parity, bin_equality = fair_metric(preds[bin_idx], bin_labels, bin_sens)
            bin_parity_mlp, bin_equality_mlp = fair_metric(preds_mlp[bin_idx], bin_labels, bin_sens)
            #print("Bin {}-{}: Parity: {}, Equality: {}".format(bin_start, bin_end, bin_parity, bin_equality))
            # get f1 score for this bin
            bin_f1 = f1_score(bin_labels, preds[bin_idx], average='binary' if self.nclass == 1 else 'micro')
            bin_f1_mlp = f1_score(bin_labels, preds_mlp[bin_idx], average='binary' if self.nclass == 1 else 'micro')
            #print("Bin {}-{}: F1: {}".format(bin_start, bin_end, bin_f1))

            f1s.append(bin_f1)
            f1s_mlp.append(bin_f1_mlp)
            parities.append(bin_parity)
            parities_mlp.append(bin_parity_mlp)
        
        f1_class_homophily_diff = np.array(f1s) - np.array(f1s_mlp)
        parity_class_homophily_diff = np.array(parities) - np.array(parities_mlp)


        f1s = []
        f1s_mlp = []
        parities = []
        parities_mlp = []
        equalities = []
        equalities_mlp = []
        # get data points with local homophily between bins of 0.1 width
        for bin_start in bins:
            bin_end = bin_start + stratify_width
            bin_idx = np.where((test_local_homophily_sens >= bin_start) & (test_local_homophily_sens < bin_end))[0]
            bin_labels = test_labels[bin_idx]
            bin_sens = test_sens[bin_idx]

            #print(len(bin_idx))
            if len(bin_idx) < 3:
                f1s.append(0)
                f1s_mlp.append(0)
                parities.append(0)
                parities_mlp.append(0)
                equalities.append(0)
                equalities_mlp.append(0)
                continue 
            bin_parity, bin_equality = fair_metric(preds[bin_idx], bin_labels, bin_sens)
            bin_parity_mlp, bin_equality_mlp = fair_metric(preds_mlp[bin_idx], bin_labels, bin_sens)
            #print("Bin {}-{}: Parity: {}, Equality: {}".format(bin_start, bin_end, bin_parity, bin_equality))
            # get f1 score for this bin
            bin_f1 = f1_score(bin_labels, preds[bin_idx], average='binary' if self.nclass == 1 else 'micro')
            bin_f1_mlp = f1_score(bin_labels, preds_mlp[bin_idx], average='binary' if self.nclass == 1 else 'micro')
            #print("Bin {}-{}: F1: {}".format(bin_start, bin_end, bin_f1))

            f1s.append(bin_f1)
            f1s_mlp.append(bin_f1_mlp)
            parities.append(bin_parity)
            parities_mlp.append(bin_parity_mlp)
            equalities.append(bin_equality)
            equalities_mlp.append(bin_equality_mlp)
        

        f1_sens_homophily_diff = np.array(f1s) - np.array(f1s_mlp)
        parity_sens_homophily_diff = np.array(parities) - np.array(parities_mlp)

        x = bins+(stratify_width/2)
        

        return f1_class_homophily_diff, parity_class_homophily_diff, f1_sens_homophily_diff, parity_sens_homophily_diff, x
    
    def eval_2D_strat(self, preds, preds_mlp, labels, eval_idx, edge_index, sens, iteration=0, model=None, power=None):

        #new_edge_index = to_undirected(edge_index)
        data = Data(edge_index=edge_index, y=labels, sens=sens, num_nodes=len(labels))

        local_homophily_class = get_local_homophily(edge_index, labels)
        local_homophily_sens = get_local_homophily(edge_index, sens)

        test_local_homophily_class = local_homophily_class[eval_idx]
        test_local_homophily_sens = local_homophily_sens[eval_idx]
        test_labels = labels[eval_idx]
        test_sens = sens[eval_idx]
        
        num_bins = 5
        perf_diff = np.zeros((num_bins, num_bins))
        parity_diff = np.zeros((num_bins, num_bins))

        for sens_bin in range(num_bins):
            for label_bin in range(num_bins):
                lower_sens = (1/num_bins) * sens_bin
                upper_sens = (1/num_bins) * (sens_bin + 1)
                lower_label = (1/num_bins) * label_bin
                upper_label = (1/num_bins) * (label_bin + 1)
                if upper_sens == 1:
                    upper_sens += 0.01
                if upper_label == 1:
                    upper_label += 0.01
                idx = np.where((test_local_homophily_sens >= lower_sens) & (test_local_homophily_sens < upper_sens) & (test_local_homophily_class >= lower_label) & (test_local_homophily_class < upper_label))[0]
                
                if len(idx) > 0:
                    # get aucroc score for this bin 
                    bin_labels = test_labels[idx]
                    bin_sens = test_sens[idx]
                    bin_preds = preds[idx]
                    bin_preds_mlp = preds_mlp[idx]

                    try: 
                        bin_f1 = f1_score(bin_labels, bin_preds, average='binary' if self.nclass == 1 else 'micro')
                        bin_f1_mlp = f1_score(bin_labels, bin_preds_mlp, average='binary' if self.nclass == 1 else 'micro')
                    except:
                        bin_f1 = 0
                        bin_f1_mlp = 0

                    try: 
                        bin_parity, bin_equality = fair_metric(bin_preds, bin_labels, bin_sens)
                        bin_parity_mlp, bin_equality_mlp = fair_metric(bin_preds_mlp, bin_labels, bin_sens)
                    except:
                        bin_parity = 0
                        bin_equality = 0
                        bin_parity_mlp = 0
                        bin_equality_mlp = 0

                    # save gnn-mlp diffs 
                    perf_diff[sens_bin, label_bin] = bin_f1 - bin_f1_mlp
                    parity_diff[sens_bin, label_bin] = bin_parity - bin_parity_mlp
        
        '''
        # plot the diffs
        plt.figure()
        plt.imshow(perf_diff.T, origin='lower')
        plt.colorbar()
        plt.title('AUCROC GNN - AUCROC MLP')
        plt.savefig('figures/local_{1}_aucroc_{2}_diff_{0}.png'.format(iteration, model, power))

        plt.figure()
        plt.imshow(parity_diff.T, origin='lower')
        plt.colorbar()
        plt.title('Parity GNN - Parity MLP')
        plt.savefig('figures/local_{1}_parity_{2}_diff_{0}.png'.format(iteration, model, power))

        plt.figure()
        plt.imshow(equality_diff.T, origin='lower')
        plt.colorbar()
        plt.title('Equality GNN - Equality MLP')
        plt.savefig('figures/local_{1}_equality_{2}_diff_{0}.png'.format(iteration, model, power))
        '''

        return perf_diff, parity_diff 

def evaluate(dataset, metric, nclass, device, iterate_over, power_iterated_over, use_2d=False):


    models = ['gcn', 'linkx', 'sage']#, 'gcnii']

    if power_iterated_over == 'class':
        class_powers = [0.0, 1.0, 3.0]
        sens_powers = [0.0]
    elif power_iterated_over == 'sens':
        class_powers = [0.0]
        sens_powers = [0.0, 1.0, 3.0]

    powers = list(itertools.product(class_powers, sens_powers))
    
    # either iterate over seeds, thus varying splits of data 
    if iterate_over == 'seed':
        seeds = [123, 246, 492]#, 246, 492]
        num_runs = 1
    # otherwise we will iterate over runs for a particular seed 
    else:
        seeds = [123] 
        num_runs = 3

    # global plots
    perf, ax_perf = plt.subplots()
    parity, ax_parity = plt.subplots()

    perf_only_mlp, ax_only_mlp = plt.subplots()
    parity_only_mlp, ax_parity_mlp = plt.subplots()

    perf_nm, ax_perf_nm = plt.subplots()
    parity_nm, ax_parity_nm = plt.subplots()

    power_f1_class = []
    power_parity_class = []
    power_f1_sens = []
    power_parity_sens = []

    for p_val in powers:
        power_f1_class.append(plt.subplots())
        power_parity_class.append(plt.subplots())
        power_f1_sens.append(plt.subplots())
        power_parity_sens.append(plt.subplots())

    for model in models:
        # get metrics for each model for global analysis  
        mean_gnn_perfs = []
        std_gnn_perfs = []
        mean_mlp_perfs = []
        std_mlp_perfs = []
        
        mean_gnn_parities = []
        std_gnn_parities = []
        mean_mlp_parities = []
        std_mlp_parities = []

        if metric != 'power':
            distances = []
        else:
            distances = powers 

        # for 1d plots across models
        f1_class_per_model = []
        f1_class_per_model_std = []
        parity_class_per_model = []
        parity_class_per_model_std = []
        f1_sens_per_model = []
        f1_sens_per_model_std = []
        parity_sens_per_model = []
        parity_sens_per_model_std = []
        x_per_model = []

        global_class = []
        global_sens = []


        # for 2d plots
        f1_2d_per_model = []
        parity_2d_per_model = []

        for power in powers: 
            print(power)
            sys.stdout.flush()
            class_power = power[0]
            sens_power = power[1]

            # will aggregate over everything in these inner loops 
            gnn_perfs = []
            mlp_perfs = []
            gnn_parities = []
            mlp_parities = []
            gnn_equalities = []
            mlp_equalities = []

            # for 1d plots 
            f1_class_data = []
            parity_class_data = []
            f1_sens_data = []
            parity_sens_data = []
            x_data = []

            class_homophily = []
            sens_homophily = []

            # for 2d plots
            f1_2d_data = []
            parity_2d_data = []

            for seed in seeds:
                gnn_save_folder = f'{dataset}_{model}stratify_class_{class_power}_sens_{sens_power}_{seed}'
                mlp_save_folder = f'{dataset}_mlpstratify_class_{class_power}_sens_{sens_power}_{seed}'
                features, labels, sens, edge_index, idx_train, idx_val, idx_test = load_dataset(dataset, device, stratify=True, \
                                                                                                class_power=class_power, sens_power=sens_power, \
                                                                                                seed=seed)

                for run in range(num_runs):

                    try:
                        gnn_evaluator = Evaluator(model, gnn_save_folder, nclass, features.shape[0], features.shape[1], iter=run)
                    except Exception as e:
                        print(e)
                        continue 
                    mlp_evaluator = Evaluator('mlp', mlp_save_folder, nclass, features.shape[0], features.shape[1], iter=run)
            
                    f1_gnn, preds_gnn = gnn_evaluator.eval_model(features, labels, idx_test, edge_index)
                    f1_mlp, preds_mlp = mlp_evaluator.eval_model(features, labels, idx_test, edge_index)
                    
                    parity_gnn, equality_gnn = fair_metric(preds_gnn, labels[idx_test].cpu().numpy(), sens[idx_test])
                    parity_mlp, equality_mlp = fair_metric(preds_mlp, labels[idx_test].cpu().numpy(), sens[idx_test])

                    f1_class, parity_class, f1_sens, parity_sens, x = gnn_evaluator.eval_1D_strat(preds_gnn, preds_mlp, labels, idx_test, edge_index, sens, dataset=dataset)
                    
                    if use_2d:
                        f1_2d_results, parity_2d_results = gnn_evaluator.eval_2D_strat(preds_gnn, preds_mlp, labels, idx_test, edge_index, sens)
                    # save global data
                    gnn_perfs.append(f1_gnn)
                    mlp_perfs.append(f1_mlp)
                    gnn_parities.append(parity_gnn)
                    mlp_parities.append(parity_mlp)
                    gnn_equalities.append(equality_gnn)
                    mlp_equalities.append(equality_mlp)

                    # save 1d data
                    f1_class_data.append(f1_class)
                    parity_class_data.append(parity_class)
                    f1_sens_data.append(f1_sens)
                    parity_sens_data.append(parity_sens)
                    x_data = x

                    # save 2d data 
                    if use_2d:
                        f1_2d_data.append(f1_2d_results)
                        parity_2d_data.append(parity_2d_results)
                    
                # if metric is something we need to compute, compute and save it 
                if metric != 'power' and seed == 123: 
                    local_homophily_class = get_local_homophily(edge_index, labels)
                    local_homophily_sens = get_local_homophily(edge_index, sens)
                    train_distr, _, _ = np.histogram2d(local_homophily_class[idx_train], local_homophily_sens[idx_train], 5)
                    test_distr, _, _ = np.histogram2d(local_homophily_class[idx_test], local_homophily_sens[idx_test], 5)

                    train_distr = train_distr / train_distr.sum()
                    test_distr = test_distr / test_distr.sum()

                    train_test_dist = dist_metric(train_distr, test_distr)
                    distances.append(train_test_dist)

            # get average and std for each metric
            gnn_perf_avg = np.mean(gnn_perfs)
            mlp_perf_avg = np.mean(mlp_perfs)
            gnn_perf_std = np.std(gnn_perfs)
            mlp_perf_std = np.std(mlp_perfs)

            gnn_parity_avg = np.mean(gnn_parities)
            mlp_parity_avg = np.mean(mlp_parities)
            gnn_parity_std = np.std(gnn_parities)
            mlp_parity_std = np.std(mlp_parities)

            gnn_equality_avg = np.mean(gnn_equalities)
            mlp_equality_avg = np.mean(mlp_equalities)
            gnn_equality_std = np.std(gnn_equalities)
            mlp_equality_std = np.std(mlp_equalities)


            mean_gnn_perfs.append(gnn_perf_avg)
            std_gnn_perfs.append(gnn_perf_std)
            mean_mlp_perfs.append(mlp_perf_avg)
            std_mlp_perfs.append(mlp_perf_std)
            mean_gnn_parities.append(gnn_parity_avg)
            std_gnn_parities.append(gnn_parity_std)
            mean_mlp_parities.append(mlp_parity_avg)
            std_mlp_parities.append(mlp_parity_std)
            
            # get aggregated 1d metrics 
            f1_class_data = np.array(f1_class_data)
            print(f1_class_data)
            f1_class_per_model.append(f1_class_data.mean(axis=0))
            f1_class_per_model_std.append(f1_class_data.std(axis=0))
            
            parity_class_data = np.array(parity_class_data)
            print(parity_class_data)
            parity_class_per_model.append(parity_class_data.mean(axis=0))
            parity_class_per_model_std.append(parity_class_data.std(axis=0))

            f1_sens_data = np.array(f1_sens_data)
            print(f1_sens_data)
            f1_sens_per_model.append(f1_sens_data.mean(axis=0))
            f1_sens_per_model_std.append(f1_sens_data.std(axis=0))

            parity_sens_data = np.array(parity_sens_data)
            parity_sens_per_model.append(parity_sens_data.mean(axis=0))
            parity_sens_per_model_std.append(parity_sens_data.std(axis=0))

            x_per_model = x_data

            # aggregate 2d results
            if use_2d:
                f1_2d_data = np.array(f1_2d_data)
                f1_2d_per_model.append(f1_2d_data.mean(axis=0))

                parity_2d_data = np.array(parity_2d_data)
                parity_2d_per_model.append(parity_2d_data.mean(axis=0))
            
        # GLOBAL PLOTS

        # plot the results
        ax_perf.errorbar(distances, np.array(mean_gnn_perfs) - np.array(mean_mlp_perfs), yerr=std_gnn_perfs, label=model, marker='o')
        #plt.errorbar(powers, mean_mlp_perfs, yerr=std_mlp_perfs, label='MLP', fmt='o')
        ax_parity.errorbar(distances, np.array(mean_gnn_parities) - np.array(mean_mlp_parities), yerr=std_gnn_parities, label=model, marker='o')
        #plt.errorbar(powers, mean_mlp_parities, yerr=std_mlp_parities, label='MLP', fmt='o')
        
        ############ JUST MLP
        # plot the results
        ax_only_mlp.errorbar(distances, np.array(mean_mlp_perfs), yerr=std_mlp_perfs, label='MLP', marker='o')
        #plt.errorbar(powers, mean_mlp_perfs, yerr=std_mlp_perfs, label='MLP', fmt='o')
        ax_parity_mlp.errorbar(distances, np.array(mean_mlp_parities), yerr=std_mlp_parities, label='MLP', marker='o')
        #plt.errorbar(powers, mean_mlp_parities, yerr=std_mlp_parities, label='MLP', fmt='o')

        ###### WITHOUT MLP SUBTRACTION ######
        # plot the results
        ax_perf_nm.errorbar(distances, np.array(mean_gnn_perfs), yerr=std_gnn_perfs, label=model, marker='o')
        #plt.errorbar(powers, mean_mlp_perfs, yerr=std_mlp_perfs, label='MLP', fmt='o')
        ax_parity_nm.errorbar(distances, np.array(mean_gnn_parities), yerr=std_gnn_parities, label=model, marker='o')
        #plt.errorbar(powers, mean_mlp_parities, yerr=std_mlp_parities, label='MLP', fmt='o')
    
        # 1D PLOTS
        for p_val in range(len(powers)):
            class_power = powers[p_val][0]
            sens_power = powers[p_val][1]

            # 1d plots
            plot_obj, axs_obj = power_f1_class[p_val]
            print(x_per_model)
            print(f1_class_per_model[p_val])
            axs_obj.errorbar(x_per_model, f1_class_per_model[p_val], yerr=f1_class_per_model_std[p_val], label=model)

            plot_obj, axs_obj = power_parity_class[p_val]
            axs_obj.errorbar(x_per_model, parity_class_per_model[p_val], yerr=parity_class_per_model_std[p_val], label=model)

            plot_obj, axs_obj = power_f1_sens[p_val]
            print(x_per_model)
            print(f1_sens_per_model[p_val])
            axs_obj.errorbar(x_per_model, f1_sens_per_model[p_val], yerr=f1_sens_per_model_std[p_val], label=model)

            plot_obj, axs_obj = power_parity_sens[p_val]
            axs_obj.errorbar(x_per_model, parity_sens_per_model[p_val], yerr=parity_sens_per_model_std[p_val], label=model)

            # 2d plots
            # do not need to pre-allocate as in 1d case as only one model per 2d plot

            if use_2d:
                plt.figure()
                plt.imshow(f1_2d_per_model[p_val].T, origin='lower')
                plt.colorbar()
                plt.title('F1 GNN - F1 MLP (Class {0}, Sens {1})'.format(class_power, sens_power))
                plt.savefig('figures/{0}_{1}/2d_f1_model_{2}_power_{3}_{4}.png'.format(dataset, metric, model, class_power, sens_power))

                plt.figure()
                plt.imshow(parity_2d_per_model[p_val].T, origin='lower')
                plt.colorbar()
                plt.title('Parity GNN - Parity MLP (Class {0}, Sens {1})'.format(class_power, sens_power))
                plt.savefig('figures/{0}_{1}/2d_parity_model_{2}_power_{3}_{4}.png'.format(dataset, metric, model, class_power, sens_power))


    if not os.path.exists('figures/{0}_{1}'.format(dataset, metric)):
        os.mkdir('figures/{0}_{1}'.format(dataset, metric))


    # save global plots
    ax_perf.set_xlabel('{0}'.format(metric))
    ax_perf.set_ylabel('GNN - MLP F1 (Micro)')
    ax_perf.legend()
    ax_perf.set_title('{0} vs F1 (Micro) - Iterate over {1}'.format(metric, power_iterated_over))
    perf.savefig('figures/{0}_{1}/perf_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))

    ax_parity.set_xlabel('{0}'.format(metric))
    ax_parity.set_ylabel('GNN Parity - MLP Parity')
    ax_parity.legend()
    ax_parity.set_title('{0} vs Parity - Iterate over {1}'.format(metric, power_iterated_over))
    parity.savefig('figures/{0}_{1}/parity_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))

    ax_only_mlp.set_xlabel('{0}'.format(metric))
    ax_only_mlp.set_ylabel('MLP F1 (Micro)')
    ax_only_mlp.legend()
    ax_only_mlp.set_title('{0} vs F1 Score - Iterate over {1}'.format(metric, power_iterated_over))
    perf_only_mlp.savefig('figures/{0}_{1}/only_mlp_perf_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))
            
    ax_parity_mlp.set_xlabel('{0}'.format(metric))
    ax_parity_mlp.set_ylabel('MLP Parity')
    ax_parity_mlp.legend()
    ax_parity_mlp.set_title('{0} vs Parity - Iterate over {1}'.format(metric, power_iterated_over))
    parity_only_mlp.savefig('figures/{0}_{1}/only_mlp_parity_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))

    ax_perf_nm.set_xlabel('{0}'.format(metric))
    ax_perf_nm.set_ylabel('GNN F1 (Micro)')
    ax_perf_nm.legend()
    ax_perf_nm.set_title('{0} vs F1 (Micro) - Iterate over {1}'.format(metric, power_iterated_over))
    perf_nm.savefig('figures/{0}_{1}/only_gnn_perf_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))

    ax_parity_nm.set_xlabel('{0}'.format(metric))   
    ax_parity_nm.set_ylabel('GNN Parity')
    ax_parity_nm.legend()
    ax_parity_nm.set_title('{0} vs Parity - Iterate over {1}'.format(metric, power_iterated_over))
    parity_nm.savefig('figures/{0}_{1}/only_gnn_parity_{2}_poweriterated_{3}.png'.format(dataset, metric, iterate_over, power_iterated_over))

    # save 1d plots
    for p_val in range(len(powers)):

        class_power, sens_power = powers[p_val]

        plot_obj, axs_obj = power_f1_class[p_val]
        axs_obj.set_ylabel('GNN - MLP F1 (Micro)')
        axs_obj.set_xlabel('Local Homophily (Class)')
        axs_obj.legend()
        axs_obj.set_title('Local Homophily (Class) vs F1 (Micro)')
        plot_obj.savefig('figures/{0}_{1}/class_stratified_f1_{2}_power_{3}_{4}.png'.format(dataset, metric, iterate_over, class_power, sens_power))


        plot_obj, axs_obj = power_parity_class[p_val]
        axs_obj.set_ylabel('GNN - MLP Parity')
        axs_obj.set_xlabel('Local Homophily (Class)')
        axs_obj.legend()
        axs_obj.set_title('Local Homophily (Class) vs Parity')
        plot_obj.savefig('figures/{0}_{1}/class_stratified_parity_{2}_power_{3}_{4}.png'.format(dataset, metric, iterate_over, class_power, sens_power))

        plot_obj, axs_obj = power_f1_sens[p_val]
        axs_obj.set_ylabel('GNN - MLP F1 (Micro)')
        axs_obj.set_xlabel('Local Homophily (Sens)')
        axs_obj.legend()
        axs_obj.set_title('Local Homophily (Sens) vs F1 (Micro)')
        plot_obj.savefig('figures/{0}_{1}/sens_stratified_f1_{2}_power_{3}_{4}.png'.format(dataset, metric, iterate_over, class_power, sens_power))

        plot_obj, axs_obj = power_parity_sens[p_val]
        axs_obj.set_ylabel('GNN - MLP Parity')
        axs_obj.set_xlabel('Local Homophily (Sens)')
        axs_obj.legend()
        axs_obj.set_title('Local Homophily (Sens) vs Parity')
        plot_obj.savefig('figures/{0}_{1}/sens_stratified_parity_{2}_power_{3}_{4}.png'.format(dataset, metric, iterate_over, class_power, sens_power))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--power_iterated_over', type=str, default='sens')

    ##### INPUTS
    # Distance Metric
    metric = 'EMD'
    if metric == 'KL':
        dist_metric = KL
    elif metric == 'JS':
        dist_metric = JS
    elif metric == 'BD':
        dist_metric = BD
    elif metric == 'EMD':
        dist_metric = emd
    elif metric == 'power':
        dist_metric = None
    
    args = parser.parse_args()
    power_iterated_over = args.power_iterated_over

    # Dataset
    for dataset in ['bail', 'tolokers', 'pokec']: #'fb100-cornell5', 'fb100-penn94', 
        print(dataset)
        # Iterate over the seed or runs
        iterate_over = 'seed'
        #iterate_over = 'runs'

        if dataset in ['tolokers', 'bail']:
            nclass = 1
        elif dataset in ['pokec', 'fb100-penn94', 'fb100-cornell5']:
            nclass = 5

        device = torch.device("cpu")

        save_folder = f'{dataset}_{metric}'

        if not os.path.exists('figures/{0}'.format(save_folder)):
            os.makedirs('figures/{0}'.format(save_folder))  

        # run evaluate function with inputs
        evaluate(dataset, metric, nclass, device, iterate_over, power_iterated_over)


    
