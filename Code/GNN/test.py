# %%
import argparse
from operator import inv
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Flickr, Reddit2, Coauthor
from model import GCN
from torch_geometric.utils import degree

# %%
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
# dataset = Planetoid(path, "pubmed", transform=T.NormalizeFeatures())

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'reddit')
# dataset = Reddit2(path, transform=T.NormalizeFeatures())

# arxiv
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToUndirected())
# data = dataset[0]
# elif args.dataset == 'flickr':
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'flickr')
# dataset = Flickr(path, transform=T.Compose([T.ToUndirected()]))
# elif args.dataset == 'reddit':
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'reddit')
dataset = Reddit2(path, transform=T.Compose([T.ToUndirected()]))
data = dataset[0]
degrees = degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
print(torch.median(degrees))

dergees= degrees.numpy()
# from utils.util import set_train_val_test_split

# data = set_train_val_test_split(2406525885, data, 5000, 20)

# %%
import plfit
from numpy.random import rand,seed

myplfit=plfit.plfit(degrees,usefortran=False)
myplfit.plfit()

# %%
from scipy.stats import pearsonr
import numpy as np
ratios = np.arange(1.0, 0.0, -0.1)
accs = np.array([
0.7902,
0.7827,
0.7718,
0.7627,
0.7617,
0.7390,
0.7371,
0.7313,
0.7329,
0.7213,
])

pearsonr(ratios, accs)

# %%
from torch_geometric.utils import to_scipy_sparse_matrix

G = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

# %%
from utils.pagerank import pagerank_scipy

pagerank_scipy(G)

# %%
from model import GCN
model = GCN(in_channels=16, 
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            dropout=0.5)

# %%
from model import GIN
model = GIN(in_channels=16, 
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            dropout=0.5,
            add_skip_connection=True)

# %%
from model import GAT
model = GAT(in_channels=16, 
            hidden_channels=16,
            out_channels=2,
            num_layers=3,
            num_heads = 3,
            dropout=0.5)

# %%
from utils.random_graphs import generate_ba_graph
from torch_geometric.utils import degree


data = generate_ba_graph(1000, 3)
degrees = degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
test_degrees = degrees[data.test_mask]
test_idxes_w_degrees = list(zip(data.test_mask, test_degrees))
test_idxes_w_degrees.sort(key=lambda x: x[-1])
test_idxes = [pair[0] for pair in test_idxes_w_degrees]
test_idxes = torch.LongTensor(test_idxes)

# %%
import torch_geometric
import numpy as np
group_num = 4
group_labels = torch.zeros_like(data.y)

if hasattr(data, "adj_t"):
    degrees = data.adj_t.sum(0)
else:
    degrees = torch_geometric.utils.degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)

metrics = degrees
def split_by_median(idxes):
    tmp_median = torch.median(metrics[idxes])
    group_1 = idxes[metrics[idxes]<=tmp_median]
    group_2 = idxes[metrics[idxes]>tmp_median]
    return group_1, group_2

group_idxes = [np.arange(metrics.size(0))]
for _ in range(group_num-1):
    tmp_idxes = group_idxes[0]
    group_1, group_2 = split_by_median(tmp_idxes)
    group_idxes.pop(0)
    group_idxes.append(group_1); group_idxes.append(group_2)
for i, idxes in enumerate(group_idxes):
    group_labels[idxes] = i

# %%
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric.loader import NeighborSampler, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree

dataset = PygNodePropPredDataset('ogbn-arxiv',transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures()]) )
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-arxiv')

transform = T.ToSparseTensor(remove_edge_index=False)
data = dataset[0]
data = transform(data)

data.adj_t = data.adj_t.to_symmetric()
degrees = data.adj_t.sum(0)
print(torch.median(degrees))
# data.n_id = torch.arange(data.num_nodes)

# train_idx = split_idx['train']

# train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
#                                sizes=[-1, -1, -1], batch_size=64,
#                                shuffle=True, num_workers=12)
# subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
#                                   batch_size=4096, shuffle=False,
                                #   num_workers=12)
# %%
from utils.util import k_hop_neighbors

k_hop_nbrs = k_hop_neighbors(data.edge_index, data.num_nodes, 2)

from models.reweighted_gcn_conv import dispersion_norm

norms = dispersion_norm(data.x, data.edge_index, k_hop_nbrs, pow_deg=-1)

# %%
from networkx.generators import barabasi_albert_graph, stochastic_block_model

node_num = 1000
sizes_list = [int(node_num*0.5), int(node_num*0.5)] 
intraclass_prob = 0.01; interclass_prob = 0.001

network = stochastic_block_model(
    sizes = sizes_list,  p = [[intraclass_prob, interclass_prob], [interclass_prob, intraclass_prob]], seed=42,
)

# %%
import numpy as np

degrees = list(network.degree)
# degrees.sort(key=lambda x: x[1])

block1_nodes = list(network.graph['partition'][0])
block2_nodes = list(network.graph['partition'][1])

labels = np.zeros(node_num)
labels[block2_nodes] = 1

# %%
def k_hop_neighbors(G, start, k):
    all_neighbors = set(); nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
        all_neighbors = all_neighbors.union(nbrs)
    return list(all_neighbors)

def ratio_of_inclass(G, labels, k):
    ratios = []
    for node_i in range(network.number_of_nodes()):
        nbrs = k_hop_neighbors(G, node_i, k)
        if len(nbrs) == 0:
            ratio = 0
        else:
            ratio = (labels[nbrs] == labels[node_i]).mean()
        ratios.append((node_i, ratio))
    return ratios

ratios = ratio_of_inclass(network, labels, 2)
# ratios.sort(key=lambda x: x[1])

# %%
from scipy.stats import spearmanr

# degrees = np.array([pair[1] for pair in degrees])
# ratios = np.array([pair[1] for pair in ratios])

print(spearmanr(degrees, ratios))

''' 
Hypothesis:
    Why is the performance in long tail nodes worse?
        Because its node feature does not progate much to the training nodes 

    why should we use SBM model?
        test the correlation between the precentage of the neighbors and node degrees
        are long tail nodes the nodes between two classes
    what is the performance of long tail nodes
    what is the precentage of the neighbors (two hop neighbors)
        does the reweighting work


1. Try directly assigns weights to the node features and search over the weights. 
'''


# %%
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Flickr, Reddit2, Coauthor
from model import GCN
from torch_geometric.utils import degree
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
# dataset = Planetoid(path, "cora", transform=T.NormalizeFeatures())
# data = dataset[0]
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'reddit')
dataset = Reddit2(path, transform=T.Compose([T.ToUndirected()]))
data = dataset[0]

# %%
import numpy as np
import scipy.sparse as sp
from utils.sample import to_edge_vertex_matrix, compute_leverage_score

B = to_edge_vertex_matrix(data.edge_index, data.num_nodes)
w = np.ones(data.num_edges)

# %%
import scipy
L = B.T @ sp.diags(w) @ B
W_sqrt = sp.diags(np.sqrt(w))

# %%
scores = compute_leverage_score(B, L.todense())

# %%
# k = int(np.ceil(np.log(B.shape[1] / epsilon**2)))
# Compute the random projection matrix
k = 400
Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
Q *= 1 / np.sqrt(k)
Y =  W_sqrt @ B
Y = Q @ Y

# %%
import time
Z = []
for i in range(5):
    start = time.time()
    Z_i = sp.linalg.minres(L, Y[i, :])[0]
    Z.append(Z_i)
    print(time.time() - start)

# %%
Z = np.stack(Z, axis=0)
leverage_scores = compute_effective_resistances(Z, data.edge_index.numpy())
print(leverage_scores.sum())

# %%

Z = Q @ W_sqrt @ B @ scipy.linalg.pinv2(L.todense())
leverage_scores = compute_effective_resistances(Z, data.edge_index.numpy())
print(leverage_scores.sum())
print(np.abs(leverage_scores-scores).max())
print(((leverage_scores-scores)<0.1).sum())
print(scipy.stats.spearmanr(leverage_scores, scores))

# %%
# Z = W_sqrt @ B @ scipy.linalg.pinv2(L.todense())
score_sums_list = []
for batch_size in [128, 256, 512]:
    Z, score_sums = compute_Z(L, W_sqrt, B, k=200, eta=1e-5, max_iters=10000, log_every=100, convergence_after = 10000, edge_index = data.edge_index, batch_size=batch_size)
    score_sums_list.append(score_sums)
# leverage_scores = compute_effective_resistances(Z, data.edge_index.numpy())
# print(((leverage_scores-scores)<0.05).sum())
# print(scipy.stats.spearmanr(leverage_scores, scores))

# %%
import matplotlib.pyplot as plt

plt.plot(score_sums_list[0])
plt.plot(score_sums_list[1])
plt.plot(score_sums_list[2])
# plt.plot(score_sums_list[3])
plt.show()

# %%

# %%
k = 100
Z = np.random.randn(k, L.shape[1])/(2*np.math.sqrt(k))
leverage_scores = compute_effective_resistances(Z, data.edge_index.numpy())
print(leverage_scores)

# %%
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def compute_Z(L, W_sqrt, B, k=100, eta=1e-3, max_iters=1000, convergence_after = 100,
                                    tolerance=1e-2, log_every=100, compute_exact_loss=False, edge_index=None, batch_size = 16):
    """ Computes the Z matrix using gradient descent.
    
    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    eta : float
        Step size for the gradient descent.
    max_iters : int
        Maximum number of iterations.
    convergence_after : int
        If the loss did not decrease significantly for this amount of iterations, the gradient descent will abort.
    tolerance : float
        The minimum amount of energy decrease that is expected for iterations. If for a certain number of iterations
        no overall energy decrease is detected, the gradient descent will abort.
    log_every : int
        Log the loss after each log_every iterations.
    compute_exact_loss : bool
        Only for debugging. If set it computes the actual pseudo inverse without down-projection and checks if
        the pairwise distances in Z's columns are the same with respect to the forbenius norm.
        
    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    # Compute the random projection matrix
    # Theoretical value of k := int(np.ceil(np.log(B.shape[1] / epsilon**2))), However the constant could be large.
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = (W_sqrt @ B).tocsr()
    Y_red = Q @ Y

    if compute_exact_loss:
        # Use exact effective resistances to track actual similarity of the pairwise distances
        L_inv = np.linalg.pinv(L.todense())
        Z_gnd = sp.csr_matrix.dot(Y, L_inv)
        pairwise_dist_gnd = Z_gnd.T.dot(Z_gnd)
    
    # Use gradient descent to solve for Z
    Z = np.random.randn(k, L.shape[1])/(1.4*np.math.sqrt(k))
    best_loss = np.inf
    best_iter = np.inf

    score_sums = []
    for it in range(max_iters):
        batch = np.random.choice(L.shape[1], batch_size, replace=False)

        residual = Y_red[:, batch] - Z @ L[:, batch]
        loss = np.linalg.norm(residual)
        if it % log_every == 0: 
            leverage_scores = compute_effective_resistances(Z, edge_index.numpy())
            print(f'Loss before iteration {it}: {loss}')
            print(f'Leverage score before iterations: {it}: {leverage_scores.sum()}')
            score_sums.append(leverage_scores.sum())
            if compute_exact_loss:
                pairwise_dist = Z.T.dot(Z)
                exact_loss = np.linalg.norm(pairwise_dist - pairwise_dist_gnd)
                print(f'Loss w.r.t. exact pairwise distances {exact_loss}')
        
        if loss + tolerance < best_loss:
            best_loss = loss
            best_iter = it
        elif it > best_iter + convergence_after:
            # No improvement for 10 iterations
            print(f'Convergence after {it - 1} iterations.')
            break
        
        Z += eta * residual @ (L[:, batch]).T # L.dot(residual.T).T
    return Z, score_sums

def compute_effective_resistances(Z, edges):
    """ Computes the effective resistance for each edge in the graph.
    
    Paramters:
    ----------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate effective resistances.
    edges : tuple
        A tuple of lists indicating the row and column indices of edges.
        
    Returns:
    --------
    R : ndarray, shape [e]
        Effective resistances for each edge.
    """
    rows, cols = edges
    assert(len(rows) == len(cols))
    R = []
    # Compute pairwise distances
    for i, j in zip(rows, cols):
        R.append(np.linalg.norm(Z[:, i] - Z[:, j]) ** 2)
    return np.array(R)

# %%
