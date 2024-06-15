
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import fair_metric_np
import warnings
from scipy.stats import beta, wasserstein_distance

warnings.filterwarnings("ignore")




all_data = {} 
use_mlp = True 
use_beta = False 
#dataset = 'fb100-penn94' 
dataset = 'pokec'
with open(f'results_synth_{dataset}_gcn_raw_results.pkl', 'rb') as f:
    gcn_data = pickle.load(f)
all_data['gcn'] = gcn_data

try:
    with open(f'results_synth_{dataset}_sage_raw_results.pkl', 'rb') as f:
        sage_data = pickle.load(f)
    all_data['sage'] = sage_data
except:
    pass 

try:
    with open(f'results_synth_{dataset}_linkx_raw_results.pkl', 'rb') as f:
        linkx_data = pickle.load(f)
    all_data['linkx'] = linkx_data
except:
    pass

#with open(f'results_synth_{dataset}_gcnii_raw_results.pkl', 'rb') as f:
#    gcnii_data = pickle.load(f)
#all_data['gcnii'] = gcnii_data



keys = all_data['gcn'].keys()
keys = ['_'.join(k.split('_')[:-1]) for k in keys]
keys = np.unique(keys)
print(keys)

num_bins = 3
width = 1/float(num_bins)
bins = np.arange(num_bins + 1) / num_bins


all_keys = [['3.0_10.0', '3.0_5.0', '5.0_3.0', '10.0_3.0'], ['2.0_5.0', '10.0_25.0', '20.0_50.0'], ['5.0_2.0', '25.0_10.0', '50.0_20.0']]
for p, keys in enumerate(all_keys): 

    fig, axs = plt.subplots(2, len(keys), sharex=True, sharey='row')

    for i, uniq_k in enumerate(keys):
        for model in ['gcn']:#,  'linkx']: #'sage', 
           
            data = all_data[model]
            a, b = uniq_k.split('_')
            a = float(a)
            b = float(b)

            perf_results = []
            fair_results = []

            for j, k in enumerate(data):
                if uniq_k in k:
                   
                    curr_data = data[k]
                    global_class = curr_data[0]
                    global_sens = curr_data[1]
                    test_local_homophily_class = np.array(curr_data[2])
                    test_local_homophily_sens = np.array(curr_data[3])
                    test_labels = np.array(curr_data[4])
                    test_sens = np.array(curr_data[5])
                    test_gnn_preds = np.array(curr_data[6])
                    test_mlp_preds = np.array(curr_data[7])

                    class_offset = np.abs(test_local_homophily_class - global_class)
                    sens_offset = np.abs(test_local_homophily_sens - global_sens)
                    total_nodes = len(class_offset)
                
                    likelihoods = beta.logpdf(test_local_homophily_class, a, b)
                    likelihoods = likelihoods / np.max(likelihoods)

                    all_accs = []
                    all_fairs = []

                    bin_order = []

                    for b_start in bins[:-1]:
                        b_end = np.round(b_start + width, 2)
                        if b_end == 1.0:
                            b_end = 1.01
                    
                        # get the indices of values with class_offset between b_start and b_end
                        if not use_beta:
                            indices = np.where((class_offset >= b_start) & (class_offset < b_end))[0]
                        else:
                            indices = np.where((likelihoods >= b_start) & (likelihoods < b_end))[0]
                        num_pts = len(indices)
                        bin_order.append((-num_pts, b_start, b_end))
                    
                    bin_order.sort()
                   
                    for _, b_start, b_end in bin_order: 

                        # get the indices of values with class_offset between b_start and b_end
                        if not use_beta:
                            indices = np.where((class_offset >= b_start) & (class_offset < b_end))[0]
                        else:
                            indices = np.where((likelihoods >= b_start) & (likelihoods < b_end))[0]

                        # get accuracy of these indices
                        gnn_acc = (test_labels[indices] == test_gnn_preds[indices]).sum() / float(len(indices))
                        # get fair_metric of these indices
                        gnn_parity, gnn_equality = fair_metric_np(test_gnn_preds[indices], test_labels[indices], test_sens[indices])

                        # get mlp accuracy of these indices
                        mlp_acc = (test_labels[indices] == test_mlp_preds[indices]).sum() / float(len(indices))
                        # get mlp fair_metric of these indices
                        mlp_parity, mlp_equality = fair_metric_np(test_mlp_preds[indices], test_labels[indices], test_sens[indices])
                        if mlp_parity == -1:
                            # set to nan
                            mlp_parity = np.nan
                        if gnn_parity == -1:
                            gnn_parity = np.nan
                        #print(acc)
                        if use_mlp:
                            all_accs.append(gnn_acc - mlp_acc)
                            all_fairs.append(gnn_parity - mlp_parity)
                        else:
                            all_accs.append(gnn_acc)
                            all_fairs.append(gnn_parity)
                  
                    perf_results.append(list(all_accs))
                    fair_results.append(list(all_fairs))
                else:
                    #print(f"Skipping {k}")
                    continue
            perf_results = np.array(perf_results)
            fair_results = np.array(fair_results)

            mean_perf = np.mean(perf_results, axis=0)
            std_perf = np.std(perf_results, axis=0)

            mean_fair = np.mean(fair_results, axis=0)
            std_fair = np.std(fair_results, axis=0)

            x = np.arange(len(mean_perf))/len(mean_perf)
        
            if model == 'gcn':
                axs[0, i].errorbar(x, mean_perf, yerr=std_perf, color='b', label='Accuracy - GCN', linestyle=':')
                axs[1, i].errorbar(x, mean_fair, yerr=std_fair, color='g', label='Fairness - GCN', linestyle=':')
            elif model == 'sage':
                axs[0, i].errorbar(x, mean_perf, yerr=std_perf, color='b', label='Accuracy - SAGE', linestyle='-.')
                axs[1, i].errorbar(x, mean_fair, yerr=std_fair, color='g', label='Fairness - SAGE', linestyle='-.')
            elif model == 'gcnii':
                axs[0, i].errorbar(x, mean_perf, yerr=std_perf, color='b', label='Accuracy - GCNII', linestyle='-.')
                axs[1, i].errorbar(x, mean_fair, yerr=std_fair, color='g', label='Fairness - GCNII', linestyle='-.')
            else:
                axs[0, i].errorbar(x, mean_perf, yerr=std_perf, color='b', label='Accuracy - LINKX', linestyle='--')
                axs[1, i].errorbar(x, mean_fair, yerr=std_fair, color='g', label='Fairness - LINKX',  linestyle='--')
   
            #axs[0, i].set_xlim([0, 1])
            axs[0, i].set_title(f"a: {a}, b: {b}")
            fig.text(0.04, 0.5, 'Stratified Accuracy/Fairness (GNN - MLP)', va='center', rotation='vertical')
            fig.text(0.5, 0.04, 'Distance from Local to Global (as computed by a/a+b)', ha='center')

    axs[0, -1].legend()
    axs[1, -1].legend()
    plt.suptitle('Semi-synthetic {0}'.format(dataset))
    file_name = f'key_{p}_dist_analysis_{dataset}'
    if not use_mlp:
        file_name += '_no_mlp'
    if use_beta:
        file_name += '_likelihood'
    plt.savefig(f'{file_name}.png')



'''
all_keys = [['3.0_10.0', '3.0_5.0', '5.0_3.0', '10.0_3.0'], ['2.0_5.0', '10.0_25.0', '20.0_50.0'], ['5.0_2.0', '25.0_10.0', '50.0_20.0']]
for p, keys in enumerate(all_keys): 

    fig, axs = plt.subplots(1, len(keys), sharex=True, sharey='row')

    for i, uniq_k in enumerate(keys):

        data = all_data['gcn']
        a, b = uniq_k.split('_')
        a = float(a)
        b = float(b)

        perf_results = []
        fair_results = []

        for j, k in enumerate(data):
            if uniq_k in k:
                
                curr_data = data[k]
                global_class = curr_data[0]
                global_sens = curr_data[1]
                test_local_homophily_class = np.array(curr_data[2])
                test_local_homophily_sens = np.array(curr_data[3])

             
                axs[i].hist(test_local_homophily_class, bins=bins, alpha=0.5, label='Local Homophily Class')
 
    plt.suptitle('Test Homophily Distributions'.format(model))
    file_name = f'homophily_dist_analysis_{p}'
    plt.savefig(f'{file_name}.png')
'''
    