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
import scipy
from sklearn.metrics import accuracy_score


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def fair_metric(pred, labels, sens, multi=False):
    '''
    compute statistical parity (SP) and equal opportunity (EO)
    '''
    #print(pred)
    #print(labels)
    #print(sens)
    #some datasets are ended as 1,2 rather than 0,1 
    if torch.max(sens) > 1:
        sens = sens - 1

    idx_s0 = (sens==0).numpy()
    #print(idx_s0)
    idx_s1 = (sens==1).numpy()
    #print(idx_s1)
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    #compute accuracy parity between sensitive attributes

    # get accuracy of labels and preds
    acc_s0 = accuracy_score(pred[idx_s0], labels[idx_s0])
    acc_s1 = accuracy_score(pred[idx_s1], labels[idx_s1])

    if multi:
        # multiclass 
        parity = abs(acc_s0-acc_s1)
    else: 
        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
  
    #equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1)).item()
    equality = 0
    return parity.item(), 0

def fair_metric_np(pred, labels, sens, multi=False):
    '''
    compute statistical parity (SP) and equal opportunity (EO)
    '''
    #print(pred)
    #print(labels)
    #print(sens)
    #some datasets are ended as 1,2 rather than 0,1 

    if len(sens) == 0:
        return -1, -1
    if np.max(sens) > 1:
        sens = sens - 1

   
    if multi:
        # multiclass 
        num_class = len(np.unique(labels))
        parities = []
        for c in range(num_class):
            # get indices with class c
            idx_c = (labels==c)
            pred_c = pred[idx_c]
            sens_c = sens[idx_c]
            # get indices for the class subset with s0 and s1
            idx_s0 = (sens_c==0)
            idx_s1 = (sens_c==1)
            # get parity for class 
            parity = abs(sum(pred_c[idx_s0])/sum(idx_s0)-sum(pred_c[idx_s1])/sum(idx_s1))
            parities.append(parity)
        parity = np.max(parities)
    else: 
        idx_s0 = (sens==0)
        #print(idx_s0)
        idx_s1 = (sens==1)
        #print(idx_s1)
        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1)).item()
  
    #equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1)).item()
    equality = 0
    return parity, 0

