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
print(is_undirected(data.edge_index))
data = T.ToUndirected()(data)

# %%
import networkx as nx

edge_list = data.edge_index.numpy()
edge_list = [(edge_list[0, i], edge_list[1, i]) for i in range(edge_list.shape[1])]
graph = nx.from_edgelist(edge_list)
# %%
from karateclub import DeepWalk

model = DeepWalk(workers=10)
model.fit(graph)
X = model.get_embedding()

# %%
import numpy as np
np.save(f"./embeddings/deepwalk/{state_name}_128.npy", X)