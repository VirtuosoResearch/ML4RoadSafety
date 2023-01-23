import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.datasets import Coauthor, Flickr, Reddit2, Planetoid, CoraFull
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_undirected
from torch_geometric.data import Data
import time

from model import GCN, SAGE, GAT, GIN


data = Planetoid(root='/tmp/Cora', name='Cora')
data = data[0]
print(data.train_mask.sum().item())
print(data.train_mask.size())