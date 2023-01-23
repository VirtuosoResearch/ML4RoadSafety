import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.nn import (MLP, GCNConv, SAGEConv, global_mean_pool, GATConv, GINConv)
from torch_sparse import SparseTensor

# from models.reweighted_gcn_conv import ReweightedGCNConv, dispersion_norm
# from models.gat_conv import GATConv
# from models.gin_conv import GINConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_classification=False, normalize=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if graph_classification:
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.pred_head = Linear(hidden_channels, out_channels)
        else:
            self.convs.append(GCNConv(hidden_channels, out_channels, normalize=normalize))

        self.dropout = dropout
        self.graph_classification = graph_classification
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        if self.graph_classification:
            self.pred_head.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None, batch=None, return_softmax = True, **kwargs):
        if self.graph_classification:
            for i, conv in enumerate(self.convs):
                x = conv(x, adj_t, edge_weight=edge_weight)
                if self.use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # Apply a readout pooling
            assert batch is not None
            x = global_mean_pool(x, batch)
            # Apply a linear classifier
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.pred_head(x)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, adj_t, edge_weight=edge_weight)
                if self.use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, adj_t, edge_weight=edge_weight)

        if return_softmax:
            return x.log_softmax(dim=-1)
        else:
            return x

class ReweightedGCN(GCN):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_classification=False, normalize=True):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, graph_classification, normalize)
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ReweightedGCNConv(in_channels, hidden_channels, normalize=normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                ReweightedGCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if graph_classification:
            self.convs.append(ReweightedGCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.pred_head = Linear(hidden_channels, out_channels)
        else:
            self.convs.append(ReweightedGCNConv(hidden_channels, out_channels, normalize=normalize))

        self.dropout = dropout
        self.graph_classification = graph_classification

    def forward(self, x, adj_t, k_hop_nbrs, batch=None, use_bn=True, **kwargs):
        
        if self.graph_classification:
            for i, conv in enumerate(self.convs):
                x = conv(x, adj_t)
                if use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # Apply a readout pooling
            assert batch is not None
            x = global_mean_pool(x, batch)
            # Apply a linear classifier
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.pred_head(x)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                if isinstance(adj_t, SparseTensor):
                    adj_t = dispersion_norm(x, adj_t, k_hop_nbrs)
                    x = conv(x, adj_t)
                else:
                    edge_weight = dispersion_norm(x, adj_t, k_hop_nbrs)
                    x = conv(x, adj_t, edge_weight=edge_weight)
                if use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if isinstance(adj_t, SparseTensor):
                adj_t = dispersion_norm(x, adj_t, k_hop_nbrs)
                x = self.convs[-1](x, adj_t)
            else:
                edge_weight = dispersion_norm(x, adj_t, k_hop_nbrs)
                x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        return x.log_softmax(dim=-1)

class ElementWiseLinear(torch.nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(1, size))
        else:
            self.weight = None
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
        num_heads, dropout, input_drop=0.0, attn_drop=0.0):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=attn_drop))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*num_heads, hidden_channels, heads=num_heads, dropout=attn_drop))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))
        self.convs.append(GATConv(hidden_channels*num_heads, out_channels, heads=1, dropout=attn_drop))
        
        self.bias_last = ElementWiseLinear(out_channels, weight=False, bias=True, inplace=True)
        self.input_drop = input_drop
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.bias_last.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None, edge_attr=None, **kwargs):
        x = F.dropout(x, p=self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        x = self.bias_last(x)
        return x.log_softmax(dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None, **kwargs):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        return x.log_softmax(dim=-1)

class IdentityModule(torch.nn.Module):
    """An identity module that outputs the input."""

    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x): 
        return x

class GIN(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.downsamples = torch.nn.ModuleList() # Downsample for skip connections
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout, batch_norm=False)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if in_channels != hidden_channels:
                downsample = Linear(in_channels, hidden_channels, bias=False)
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(IdentityModule())
            in_channels = hidden_channels
        mlp = MLP([hidden_channels, hidden_channels, out_channels], dropout=dropout, batch_norm=False)
        self.convs.append(GINConv(nn=mlp, train_eps=False))

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None, **kwargs):
        for i, conv in enumerate(self.convs[:-1]):
            out = conv(x, adj_t, edge_weight=edge_weight)
            out = self.bns[i](out)
            x = F.relu(out) + self.downsamples[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_weight=edge_weight)

        return x.log_softmax(dim=-1)

class ClusterGCN(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClusterGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, **kwargs):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, adj_t)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x.log_softmax(dim=-1)