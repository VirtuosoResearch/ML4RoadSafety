# %%
import os
import torch
import pandas as pd
from torch_geometric.data import Data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--state_name", type=str, default="IA")
args = parser.parse_args()

data_dir = "./data"
state_name = args.state_name

adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
edge_index = adj.coalesce().indices()
data = Data(edge_index=edge_index)

# %%
import torch.nn.functional as F
from torch_geometric.utils import degree
num_nodes = data.num_nodes
row, col = edge_index
out_deg = degree(row, num_nodes)
out_deg = out_deg.view((-1, 1))

in_deg = degree(col, num_nodes)
in_deg = in_deg.view((-1, 1))

max_deg = torch.max(out_deg.max(), in_deg.max())

# %%
in_deg_capped = torch.min(in_deg, max_deg).type(torch.int64)
in_deg_onehot = F.one_hot(
    in_deg_capped.view(-1), num_classes=int(max_deg.item()) + 1)
in_deg_onehot = in_deg_onehot.type(in_deg.dtype)

out_deg_capped = torch.min(out_deg, max_deg).type(torch.int64)
out_deg_onehot = F.one_hot(
    out_deg_capped.view(-1), num_classes=int(max_deg.item()) + 1)
out_deg_onehot = out_deg_onehot.type(out_deg.dtype)

# %%
import networkx as nx

G = nx.Graph(data.edge_index.numpy().T.tolist())
G.add_nodes_from(range(data.num_nodes))  # in case missing node ids


# %%
edge_feature_dict = torch.load(os.path.join(data_dir, f"{state_name}/Edges/edge_features.pt"))

edge_lengths = edge_feature_dict['length'].coalesce().values()
length_mean = edge_lengths[~torch.isnan(edge_lengths)].mean()
edge_lengths[torch.isnan(edge_lengths)] = length_mean

edge_length_dict = {}
for k in range(edge_index.shape[1]):
    i, j = edge_index[:, k]
    if k >= len(edge_lengths):
        edge_length_dict[(i.item(), j.item())] = length_mean
    else:
        edge_length_dict[(i.item(), j.item())] = edge_lengths[k].item()

nx.set_edge_attributes(G, edge_length_dict, "length")

# %%
# G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])

# closeness = nx.algorithms.closeness_centrality(G, distance="length")
# pagerank = nx.pagerank_numpy(G, weight="length")
# betweenness = nx.algorithms.betweenness_centrality(G)
# %%
betweenness = nx.algorithms.betweenness_centrality(G, weight="length", k=1000)
centrality_features = torch.tensor(
    [betweenness[i] for i in range(
        data.num_nodes)]).view(-1, 1) # closeness[i], pagerank[i]

# %%
x = torch.cat([in_deg, out_deg, centrality_features, in_deg_onehot, out_deg_onehot], -1)

# %%
import numpy as np
np.save(f"./embeddings/centrality/{state_name}_{int(max_deg.item())}.npy", x.numpy())