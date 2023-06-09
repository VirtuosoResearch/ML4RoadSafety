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

class GraphSAGEConv(MessagePassing):
    
    def __init__(self, in_channels_node, in_channels_edge, out_channels, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.linear_node = torch.nn.Linear(in_channels_node, out_channels)
        self.linear_edge = torch.nn.Linear(in_channels_edge, out_channels)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        # add features corresponding to self-loop edges.
        if edge_attr is not None:
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.linear_edge(edge_attr)
        else:
            edge_embeddings = None

        x = self.linear_node(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if edge_attr is not None else x_j

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)
    
class GINConv(MessagePassing):

    def __init__(self, in_channels_node, in_channels_edge, out_channels, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(out_channels, 2*out_channels), torch.nn.ReLU(), torch.nn.Linear(2*out_channels, out_channels))
        self.linear_node = torch.nn.Linear(in_channels_node, out_channels)
        self.linear_edge = torch.nn.Linear(in_channels_edge, out_channels)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        # add features corresponding to self-loop edges.
        if edge_attr is not None:
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.linear_edge(edge_attr)
        else:
            edge_embeddings = None
        x = self.linear_node(x) if x.shape[1] != self.linear_node.out_features else x
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if edge_attr is not None else x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)
    
class GATConv(MessagePassing):
    
    def __init__(self, in_channels_node, in_channels_edge, out_channels, aggr = "add", heads=2, negative_slope=0.2,):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = out_channels
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(in_channels_node, heads * out_channels)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.linear_edge = torch.nn.Linear(in_channels_edge, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        # add self loops in the edge space
        num_nodes = x.size(0) if isinstance(x, Tensor) else x[0].size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes = num_nodes)

        # add features corresponding to self-loop edges.
        if edge_attr is not None:
            self_loop_attr = torch.zeros(num_nodes, edge_attr.size(1))
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.linear_edge(edge_attr)
        else:
            edge_embeddings = None

        if isinstance(x, Tensor):
            x = self.weight_linear(x)
        else:
            x_src, x_dst = x
            x_src = self.weight_linear(x_src)
            x_dst = self.weight_linear(x_dst)
            x = (x_src, x_dst)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    # def message(self, x_i, x_j, edge_attr, edge_index_j):
    #     x_i = x_i.view(-1, self.heads, self.emb_dim)
    #     x_j = x_j.view(-1, self.heads, self.emb_dim)
    #     if edge_attr is not None:
    #         edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
            
    #         x_j += edge_attr

    #     alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
    #     alpha = F.leaky_relu(alpha, self.negative_slope)
    #     alpha = softmax(alpha, edge_index_j)

    #     return x_j * alpha.view(-1, self.heads, 1)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out