from process_data import days_with_collision

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.datasets import Coauthor, Flickr, Reddit2, Planetoid, CoraFull
from torch_geometric.data import Data
import numpy as np
from model.model import *

import argparse

def adj2edge_index(adj):
    edge_index = []
    edge_weight = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                edge_index.append([i,j])
                edge_weight.append(adj[i][j])
    return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_weight)


def dataset2graph_signal(x, y, adj):
    adj_mx = np.load(adj)
    edge_index, edge_weight = adj2edge_index(adj_mx)
    
    features = x
    targets = y
    
    return StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)



# ------------------ Data ------------------
loc = 'la'

if loc == 'la':
    x = np.load('METR-LA/node_values.npy')
    y = np.load('METR-LA/accident_data.npy')
    adj = 'METR-LA/adj_mat.npy'
elif loc == 'bay':
    x = np.load('PEMS-BAY/pems_node_values.npy')
    y = np.load('PEMS-BAY/accident_data.npy')
    adj = 'PEMS-BAY/pems_adj_mat.npy'
    
    
days = days_with_collision(loc, 1, 10)
x = x[days]
y = y[days]

# dataset = dataset2graph_signal(x_new, y_new, adj)

x = x.reshape(-1, 2)
y = y.reshape(-1, 1)

adj_mx = np.load(adj)
edge_index, weight = adj2edge_index(adj_mx)

# concatenate for a 74 days graph's edge index
new_edge_index = np.copy(edge_index)
for i in range(73):
    new_edge_index = np.concatenate((new_edge_index, edge_index+(i+1)*207), axis=1)

# a = np.concatenate((edge_index, edge_index+207), axis=1)

# # 207*74 total nodes
# # leave last two days for test
# test_mask = np.zeros((x.shape[0], 1))
# test_mask[64*207:] = 1
# # print(np.sum(test_mask))
# test_mask = torch.tensor(test_mask)
# test_mask = test_mask.bool().int()
# test_mask = test_mask.reshape(-1)

# train_mask = np.zeros((x.shape[0], 1))
# print('/------------------/')
# train_mask = y == 1
# train_mask[64*207:] = 0
# # append 0 to train mask in the behind
# train_mask = torch.tensor(train_mask).reshape(-1)

# # random set 858 entries in train mask to true
# neg_index = np.where(train_mask[:64*207] == 0)[0]
# neg_index = list(neg_index)
# # print(neg_index)


# import random
# index = random.sample(neg_index, 2000)
# index = torch.tensor(index)
# train_mask[index] = 1

# test_mask = test_mask.bool()


# x = torch.tensor(x)
# x = x.to(torch.float32)
# y = torch.tensor(y).reshape(-1)
# y = y.to(torch.int64)
# new_edge_index = torch.tensor(new_edge_index)

# data = Data(x=x, y=y, edge_index=new_edge_index, edge_weight=weight, train_mask=train_mask, test_mask=test_mask)


import torch
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.transforms import OneHotDegree

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 32)
        self.hidden = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.hidden(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
   

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True).jittable()

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        # x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default='la')
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--neg_samples', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)
    
    args = parser.parse_args()
    
    accs = []
    fscores = []
    ensemble_rate = args.ensemble
    for _ in range(args.runs):
        preds = []
        for i in range(ensemble_rate):
            # 207*74 total nodes
            # leave last two days for test
            test_mask = np.zeros((x.shape[0], 1))
            test_mask[64*207:] = 1
            # print(np.sum(test_mask))
            test_mask = torch.tensor(test_mask)
            test_mask = test_mask.bool().int()
            test_mask = test_mask.reshape(-1)

            train_mask = np.zeros((x.shape[0], 1))
            # print('/------------------/')
            train_mask = y == 1
            train_mask[64*207:] = 0
            # append 0 to train mask in the behind
            train_mask = torch.tensor(train_mask).reshape(-1)

            # random set 858 entries in train mask to true
            neg_index = np.where(train_mask[:64*207] == 0)[0]
            neg_index = list(neg_index)
            # print(neg_index)


            import random
            index = random.sample(neg_index, args.neg_samples) # switch to 200 for GIN
            index = torch.tensor(index)
            train_mask[index] = 1

            test_mask = test_mask.bool()


            x = torch.tensor(x)
            x = x.to(torch.float32)
            y = torch.tensor(y).reshape(-1)
            y = y.to(torch.int64)
            new_edge_index = torch.tensor(new_edge_index)

            data = Data(x=x, y=y, edge_index=new_edge_index, edge_weight=weight, train_mask=train_mask, test_mask=test_mask)
            # -------------------data preprocess-------------------
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = YOURMODEL
            data = data.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            model.train()
            for epoch in range(500):
                optimizer.zero_grad()
                out = model(data)
                # out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            model.eval()
            pred = model(data).argmax(dim=1)
            preds.append(pred)
            
        preds = torch.stack(preds, dim=0)
        best_pred = preds.mode(dim=0)[0]
        # print(best_pred.shape)
        # print(pred)
        # # correct until now
        # print("test mask", data.test_mask)
        # print(model.parameters())
        correct = (best_pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        # print(int(data.test_mask.sum()))
        print(f'Accuracy: {acc:.4f}')
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for i in range(best_pred[data.test_mask].shape[0]):
            p = int(best_pred[data.test_mask][i])
            r = int(data.y[data.test_mask][i])
            if p == r == 1:
                true_positive += 1
            if p != r:
                if p == 0:
                    false_negative += 1
                else:
                    false_positive += 1
        # print('True positive: {}'.format(true_positive))
        # print('False positive: {}'.format(false_positive))
        # print('False negative: {}'.format(false_negative))
        try:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            continue
        f_score = 2 * precision * recall / (precision + recall)
        print('F1-score: {:.4f}'.format(f_score))
        accs.append(acc)
        fscores.append(f_score)
    print('Accuracy: {:.4f}±{:.4f}'.format(np.mean(accs), np.std(accs)))
    print('F1-score: {:.4f}±{:.4f}'.format(np.mean(fscores), np.std(fscores)))