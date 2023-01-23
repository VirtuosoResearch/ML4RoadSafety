from process_data import days_with_collision

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

import numpy as np


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

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.hidden = DCRNN(32,32,1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.hidden(h, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h.log_softmax(dim=-1)
    


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

x_new = []
y_new = []
for i in x:
    x_new.append(i)
for i in y:
    y_new.append(i)

dataset = dataset2graph_signal(x_new, y_new, adj)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

from tqdm import tqdm

model = RecurrentGCN(node_features = 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()


# TODO: add a classfication critiria to the model and print out accuracy

model.eval()
cost = 0
correct = 0
total = 0
true_postive = 0
false = 0
y_total = 0
error = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight)

    for i in range(len(y_hat)):
        total += 1
        y_total += snapshot.y[i]
        error += abs(y_hat[i, 0]-snapshot.y[i])
        if abs(y_hat[i, 0]-snapshot.y[i]) < 0.5:
            correct += 1
            if y_hat[i, 0] > 0.5:
                true_postive += 1
        else:
            false += 1/2
print('error: {}'.format(error/total))
print('y_total: {}'.format(y_total))   
print('F1-score: {}'.format(true_postive/(true_postive+false)))
print('Number of correct predictions: {}'.format(correct))
print('Number of total predictions: {}'.format(total))
print('Accuracy: {:.4f}'.format(correct/total))
# print("MSE: {:.4f}".format(cost))
