# %%
import os
import torch
import pandas as pd

data_dir = "./data"
state_name = "MA"

adj = torch.load(os.path.join(data_dir, f"{state_name}/adj_matrix.pt"))
# n = 285942 and m = 706402
edge_index = adj.coalesce().indices()

# %%
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data
import torch_geometric.transforms as T

data = Data(edge_index=edge_index)
# print(is_undirected(data.edge_index))
# data = T.ToUndirected()(data)

# %%
import numpy as np
import networkx as nx

edge_list_numpy = data.edge_index.numpy()
edge_list = [(edge_list_numpy[0, i], edge_list_numpy[1, i]) for i in range(edge_list_numpy.shape[1])]
graph = nx.from_edgelist(edge_list)
graph.add_nodes_from(list(range(data.num_nodes)))

graph.number_of_nodes(), graph.number_of_edges()
# %%
from karateclub import Node2Vec

model = Node2Vec(workers=10)
model.fit(graph)
X = model.get_embedding()

# %%
import numpy as np
np.save(f"./embeddings/node2vec/{state_name}_128.npy", X)