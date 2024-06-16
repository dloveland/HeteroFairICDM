import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import argparse
import time
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv, SAGEConv
import types



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        # self.gc1 = spectral_norm(GCNConv(nfeat, nhid).lin)
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, edge_index, x):
        x = self.gc1(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()
        
        self.gc1 = SAGEConv(nfeat, nhid, normalize=True)

    def forward(self, edge_index, x):
        
        x = self.gc1(x, edge_index)
        return x

def accuracy(output, labels):
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_model(nfeat, args, model_name):
    if model_name == 'gcn':  
        model = GCN(nfeat, nhid=args.num_hidden, dropout=args.dropout)
    else:
        model = SAGE(nfeat, nhid=args.num_hidden, dropout=args.dropout)
    return model


class FairGNN(nn.Module):
    def __init__(
        self, nfeat, model=None, epoch=2000, nclass=1, seed=1):
        super(FairGNN, self).__init__()


        args = types.SimpleNamespace()
        args.batch_size = 100
        args.num_hidden = 128
        args.alpha = 0.001
        args.beta = 10
        args.weight_decay = 1e-5
        args.lr = 0.001
        args.dropout = 0.5
        args.acc = args.roc = 0
        args.epochs = epoch
        self.n_class = 5
        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat, 1, dropout)
        self.GNN = get_model(nfeat, args, model)
        self.classifier = nn.Linear(nhid, self.n_class)
        self.adv = nn.Linear(nhid, 1)

        G_params = (
            list(self.GNN.parameters())
            + list(self.classifier.parameters())
            + list(self.estimator.parameters())
        )
        self.optimizer_G = torch.optim.Adam(
            G_params, lr=args.lr, weight_decay=args.weight_decay
        )
        self.optimizer_A = torch.optim.Adam(
            self.adv.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.args = args
        self.sens_criterion = nn.BCEWithLogitsLoss()
        if self.n_class == 1:
            self.pred_criterion = nn.BCEWithLogitsLoss()
        else:
            self.pred_criterion = nn.CrossEntropyLoss()
        self.G_loss = 0
        self.A_loss = 0

    def fair_metric(self, sens, labels, output, idx):
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        parity = abs(
            sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1)
        )
        equality = abs(
            sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
        )

        return parity, equality

    def fair_metric_direct(self, pred, labels, sens, multi=False):
        
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
                if sum(idx_s0) == 0 or sum(idx_s1) == 0:
                    parities.append(0)
                else: 
                    parity = abs(sum(pred_c[idx_s0]==c)/sum(idx_s0)-sum(pred_c[idx_s1]==c)/sum(idx_s1))
                    parities.append(parity)
            parities = np.array(parities)
            parities = parities[~np.isnan(parities)]
            
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

    def forward(self, g, x):
        s = self.estimator(g, x)
        z = self.GNN(g, x)
        y = self.classifier(z)
        return y, s

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train, edge_index):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(edge_index, x)
        h = self.GNN(edge_index, x)
        y = self.classifier(h)

        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov = torch.abs(
            torch.mean(
                (s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))
            )
        )

        self.cls_loss = self.pred_criterion(
            y[idx_train], labels[idx_train]
        )

        self.adv_loss = self.sens_criterion(s_g, s_score)

        self.G_loss = (
            self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        )
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.sens_criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

    def fit(
        self,
        g: torch.Tensor = None,
        features: torch.Tensor = None,
        labels: torch.Tensor = None,
        idx_train: torch.Tensor = None,
        idx_val: torch.Tensor = None,
        idx_test: torch.Tensor = None,
        sens: torch.Tensor = None,
        idx_sens_train: torch.Tensor = None,
        device="cpu",
    ):
        # with args
        if idx_sens_train is None:
            idx_sens_train = idx_train
        self = self.to(device)
        features = features.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)
        sens = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

        args = self.args
        t_total = time.time()
        best_fair = 1000
        best_acc = 0

        self.g = g
        self.x = features
        self.labels = labels
        self.sens = sens

        self.edge_index = g
        self.val_loss = 0
        for epoch in tqdm(range(args.epochs)):
            t = time.time()
            self.train()
            self.optimize(
                g, features, labels, idx_train, sens, idx_sens_train, self.edge_index
            )
            self.eval()

            output, s = self(self.edge_index, features)
            if self.n_class == 1:
                # Binary Cross-Entropy
                preds = (output.squeeze()>0).type_as(labels)
            else:
                # Cross-Entropy
                preds = torch.argmax(output, axis=1)

            acc_val = accuracy(preds[idx_val], labels[idx_val])

        
            if acc_val > args.acc or epoch == 0:
                #if parity_val + equality_val < best_fair:
                if acc_val>best_acc:
                    best_epoch = epoch
                    best_acc = acc_val
                    self.val_loss = -acc_val.detach().cpu().item()
                    self.eval()
                    output, s = self.forward(self.edge_index, self.x)

                    if self.n_class == 1:
                        # Binary Cross-Entropy
                        output = (output.squeeze()>0).type_as(labels)
                    else:
                        # Cross-Entropy
                        output = torch.argmax(output, axis=1)
                    F1 = f1_score(
                        self.labels[idx_test].detach().cpu().numpy(),
                        output[idx_test],
                        average="micro" if self.n_class == 5 else 'binary',
                    )
                    ACC = accuracy_score(
                        self.labels[idx_test].detach().cpu().numpy(),
                        output[idx_test],
                    )
                    if self.labels.max() > 1:
                        AUCROC = 0
                    else:
                        try:
                            AUCROC = roc_auc_score(
                                self.labels[idx_test].detach().cpu().numpy(),
                                output[idx_test],
                            )
                        except:
                            AUCROC = 'N/A'
                    (
                        ACC_sens0,
                        AUCROC_sens0,
                        F1_sens0,
                        ACC_sens1,
                        AUCROC_sens1,
                        F1_sens1,
                    ) = self.predict_sens_group(output[idx_test], idx_test)
                    SP, EO = self.fair_metric_direct(
                        output[idx_test],
                        self.labels[idx_test].detach().cpu().numpy(),
                        self.sens[idx_test].detach().cpu().numpy(),
                        multi=True if self.n_class == 5 else False 
                    )

                    self.temp_result = (
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
                    )

            if epoch <= 10 and acc_val > args.acc:
                args.acc = acc_val

        print("Optimization Finished! Best Epoch:", best_epoch)
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def predict(self, idx_test):
        return self.temp_result

    def predict_(self, idx_test):
        self.eval()
        output, s = self.forward(self.edge_index, self.x)

        output = (output > 0).long().detach().cpu().numpy()
        F1 = f1_score(
            self.labels[idx_test].detach().cpu().numpy(),
            output[idx_test],
            average="micro",
        )
        ACC = accuracy_score(
            self.labels[idx_test].detach().cpu().numpy(),
            output[idx_test],
        )
        if self.labels.max() > 1:
            AUCROC = 0
        else:
            try:
                AUCROC = roc_auc_score(
                    self.labels[idx_test].detach().cpu().numpy(), output[idx_test]
                )
            except:
                AUCROC = 'N/A'
        (
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
        ) = self.predict_sens_group(output[idx_test], idx_test)
        SP, EO = self.fair_metric_direct(
            output[idx_test],
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
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
        )

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][
                    self.sens[idx_test].detach().cpu().numpy() == sens
                ]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test].detach().cpu().numpy() == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][
                    self.sens[idx_test].detach().cpu().numpy() == sens
                ]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test].detach().cpu().numpy() == sens],
            )
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                try:
                    AUCROC = roc_auc_score(
                        self.labels[idx_test][
                            self.sens[idx_test].detach().cpu().numpy() == sens
                        ]
                        .detach()
                        .cpu()
                        .numpy(),
                        pred[self.sens[idx_test].detach().cpu().numpy() == sens],
                    )
                except:
                    AUCROC = 'N/A'
            result.extend([ACC, AUCROC, F1])

        return result