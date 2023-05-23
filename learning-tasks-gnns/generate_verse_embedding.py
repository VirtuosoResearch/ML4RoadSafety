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
from verse.python.wrapper import VERSE

verse = VERSE(cpath="/home/ldy/verse/src")

import numpy as np
from scipy.sparse import csr_matrix

edge_index = data.edge_index.numpy()
edges = np.ones(edge_index.shape[1])

graph = csr_matrix((edges, (edge_index[0], edge_index[1])))

dim = 128
w = verse.verse_ppr(graph, n_hidden=dim)
np.save(f"./embeddings/verse/{state_name}_ppr_{dim}.npy", w)