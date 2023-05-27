import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch import Tensor

class GCNConv(MessagePassing):

    def __init__(self, in_channels_node, in_channels_edge, out_channels, aggr = "add"):
        super(GCNConv, self).__init__()

        self.linear_node = torch.nn.Linear(in_channels_node, out_channels)
        self.linear_edge = torch.nn.Linear(in_channels_edge, out_channels)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        # add features corresponding to self-loop edges.
        if edge_attr is not None:
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.linear_edge(edge_attr)
        else:
            edge_embeddings = None

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr) if edge_attr is not None else norm.view(-1, 1) * x_j
