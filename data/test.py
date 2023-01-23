#%%
import numpy as np
import pandas as pd
import sqlite3
import argparse
from datetime import datetime, timedelta, date

node_values = np.load('METR-LA/node_values.npy')
print(node_values.shape[0])
tmp = np.ones((node_values.shape[0], node_values.shape[1],1))
# concat node_values and tmp
concat = np.concatenate((node_values, tmp), axis=2)
print(concat.shape)
# %%
adj = np.load('METR-LA/adj_mat.npy')
print(adj)
print(adj.shape)
import pickle
# load pickle file
adj_com = pickle.load(open('sensor_graph/adj_mx.pkl', 'rb'))
print(len(adj_com))
print(adj_com[2])
m = adj_com[2] - adj
print(np.sum(np.abs(m)))
# %%
adj = np.load('PEMS-BAY/pems_adj_mat.npy')
print(adj)
print(adj.shape)
import pickle

# load pickle file
adj_com = pickle.load(open('sensor_graph/adj_mx_bay.pkl', 'rb'))
print(len(adj_com))
print(adj_com[2])
m = adj_com[2] - adj
print(np.sum(np.abs(m)))
# %%
import networkx as nx

# build a graph from adj matrix
adj = np.load('METR-LA/adj_mat.npy')
print(adj)
g = nx.from_numpy_matrix(adj)

# %%
import pandas as pd
# df = pd.read_hdf('metr-la/metr-la.h5')
df = pd.read_hdf('pems-bay/pems-bay.h5')

print(df.index)
# %%
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
dataset
# %%
