import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    DeepGraphInfomax,
    JumpingKnowledge,
)
import os

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert

import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np
import scipy.sparse as sp


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x




class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = "mean"
        self.transition = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = "mean"

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x





class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, base_model="gcn", k: int = 2
    ):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == "gcn":
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == "gin":
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == "sage":
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == "infomax":
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == "jk":
            self.conv = JK(in_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        return x


class NIFTY(torch.nn.Module):
    def __init__(
        self,
        adj,
        features,
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
        encoder="gcn",
        sim_coeff=0.6,
        nclass=1,
        device="cpu",
    ):
        super(NIFTY, self).__init__()

        self.device = device

        # self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        self.edge_index = adj
        self.nclass = nclass
        self.encoder = Encoder(
            in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder
        ).to(device)
        # model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff,
        # nclass=num_class).to(device)
        self.val_edge_index_1 = dropout_adj(
            self.edge_index.to(device), p=drop_edge_rate_1
        )[0]
        self.val_edge_index_2 = dropout_adj(
            self.edge_index.to(device), p=drop_edge_rate_2
        )[0]
        self.val_x_1 = drop_feature(
            features.to(device), drop_feature_rate_1, sens_idx, sens_flag=False
        )
        self.val_x_2 = drop_feature(features.to(device), drop_feature_rate_2, sens_idx)

        self.sim_coeff = sim_coeff
        # self.encoder = encoder
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx
        self.drop_edge_rate_1 = self.drop_edge_rate_2 = 0
        self.drop_feature_rate_1 = self.drop_feature_rate_2 = 0

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass)

        for m in self.modules():
            self.weights_init(m)

        par_1 = (
            list(self.encoder.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
            + list(self.fc4.parameters())
        )
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        self.optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
        self.optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
        self = self.to(device)

        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (
            -torch.max(F.softmax(x2), dim=1)[0]
            * torch.log(torch.max(F.softmax(x1), dim=1)[0])
        ).mean()

    def D(self, x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c1 = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx]) / 2
        l2 = self.D(h2[idx], p1[idx]) / 2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff * (l1 + l2), l3

    def forwarding_predict(self, emb):

        # projector
        p1 = self.projection(emb)

        # predictor
        h1 = self.prediction(p1)

        # classifier
        c1 = self.classifier(emb)

        return c1


    def ssf_validation(self, x_1, edge_index_1, x_2, edge_index_2, y):
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        l1 = self.D(h1[self.idx_val], p2[self.idx_val]) / 2
        l2 = self.D(h2[self.idx_val], p1[self.idx_val]) / 2
        sim_loss = self.sim_coeff * (l1 + l2)

        # classifier
        c1 = self.classifier(z1)
        c2 = self.classifier(z2)

        # # TODO WORK FOR BOTH
        if self.nclass == 1:
            l3 = (
                F.binary_cross_entropy_with_logits(
                    c1[self.idx_val], y[self.idx_val].unsqueeze(1).float().to(self.device)
                )
                / 2
            )
            l4 = (
                F.binary_cross_entropy_with_logits(
                    c2[self.idx_val], y[self.idx_val].unsqueeze(1).float().to(self.device)
                )
                / 2
            )
        else: 
            l3 = (
                F.cross_entropy(
                    c1[self.idx_val], y[self.idx_val].to(self.device)
                )
                / 2
            )
            l4 = (
                F.cross_entropy(
                    c2[self.idx_val], y[self.idx_val].to(self.device)
                )
                / 2
            )

        return sim_loss, l3 + l4


    def fit(self, epochs=300):

        # Train model
        t_total = time.time()
        best_loss = 100
        best_acc = 0

        for epoch in range(epochs + 1):
            t = time.time()

            sim_loss = 0
            cl_loss = 0
            rep = 1
            for _ in range(rep):
                self.train()
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                edge_index_1 = dropout_adj(self.edge_index, p=self.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(self.edge_index, p=self.drop_edge_rate_2)[0]
                x_1 = drop_feature(
                    self.features,
                    self.drop_feature_rate_1,
                    self.sens_idx,
                    sens_flag=False,
                )
                x_2 = drop_feature(
                    self.features, self.drop_feature_rate_2, self.sens_idx
                )
                z1 = self.forward(x_1, edge_index_1)
                z2 = self.forward(x_2, edge_index_2)

                # projector
                p1 = self.projection(z1)
                p2 = self.projection(z2)

                # predictor
                h1 = self.prediction(p1)
                h2 = self.prediction(p2)

                l1 = self.D(h1[self.idx_train], p2[self.idx_train]) / 2
                l2 = self.D(h2[self.idx_train], p1[self.idx_train]) / 2
                sim_loss += self.sim_coeff * (l1 + l2)

            (sim_loss / rep).backward()
            self.optimizer_1.step()

            # classifier
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            c1 = self.classifier(z1)
            c2 = self.classifier(z2)

            # TODO MAKE IT WORK FOR BOTH 
            if self.nclass == 1:
                l3 = (
                    F.binary_cross_entropy_with_logits(
                        c1[self.idx_train],
                        self.labels[self.idx_train].unsqueeze(1).float().to(self.device),
                    )
                    / 2
                )
                l4 = (
                    F.binary_cross_entropy_with_logits(
                        c2[self.idx_train],
                        self.labels[self.idx_train].unsqueeze(1).float().to(self.device),
                    )
                    / 2
                )
            else: 
                l3 = (
                    F.cross_entropy(
                        c1[self.idx_train],
                        self.labels[self.idx_train].to(self.device),
                    )
                    / 2
                )
                l4 = (
                    F.cross_entropy(
                        c2[self.idx_train],
                        self.labels[self.idx_train].to(self.device),
                    )
                    / 2
                )


            

            cl_loss = (1 - self.sim_coeff) * (l3 + l4)
            cl_loss.backward()
            self.optimizer_2.step()
            loss = sim_loss / rep + cl_loss

            # Validation
            self.eval()
            val_s_loss, val_c_loss = self.ssf_validation(
                self.val_x_1,
                self.val_edge_index_1,
                self.val_x_2,
                self.val_edge_index_2,
                self.labels,
            )
            emb = self.forward(self.val_x_1, self.val_edge_index_1)
            output = self.forwarding_predict(emb)
            
            # if epoch % 100 == 0:
            #     print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

            if (val_c_loss + val_s_loss) < best_loss:
                self.val_loss = val_c_loss.item() + val_s_loss.item()

                # print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                best_loss = val_c_loss + val_s_loss
                

    def predict(self):

 
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb)

        # Report
        if self.nclass == 1:
            # Binary Cross-Entropy
            output_preds = (output.squeeze()>0).type_as(self.labels)
        else:
            # Cross-Entropy
            output_preds = torch.argmax(output, axis=1)


        parity, equality = self.fair_metric(
            output_preds[self.idx_test].cpu().numpy(),
            self.labels[self.idx_test].cpu().numpy(),
            self.sens[self.idx_test].numpy(),
            multi=True if self.nclass == 5 else False 
        )
        f1_s = f1_score(
            self.labels[self.idx_test].cpu().numpy(),
            output_preds[self.idx_test].cpu().numpy(),
            average="micro" if self.nclass == 5 else 'binary',
        )


        return (
            f1_s,
            parity,
            equality,
        )
        # print report
        # print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        # print(f'Parity: {parity} | Equality: {equality}')
        # print(f'F1-score: {f1_s}')
        # print(f'CounterFactual Fairness: {counterfactual_fairness}')
        # print(f'Robustness Score: {robustness_score}')

    def fair_metric(self, pred, labels, sens, multi=False):

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

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
            )
            try:
                AUCROC = roc_auc_score(
                    self.labels[idx_test][self.sens[idx_test] == sens]
                    .detach()
                    .cpu()
                    .numpy(),
                    pred[self.sens[idx_test] == sens],
                )
            except:
                AUCROC = "N/A"
            result.extend([ACC, AUCROC, F1])

        return result


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1 - x[:, sens_idx]

    return x