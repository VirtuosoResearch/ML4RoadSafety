import torch
import torch.nn.functional as F
from layers import GCNConv, GraphSAGEConv, GATConv, GINConv

class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_index, edge_attr):
        return x

class GNN(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers, dropout = 0, JK = "last", gnn_type = "gcn"):
        super(GNN, self).__init__()
        self.num_layer = num_layers
        self.drop_ratio = dropout
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                tmp_in_channels_node = in_channels_node
            else:
                tmp_in_channels_node = hidden_channels
            
            if gnn_type == "gcn":
                self.gnns.append(GCNConv(tmp_in_channels_node, in_channels_edge, hidden_channels))
            elif gnn_type == "gin":
                self.gnns.append(GINConv(tmp_in_channels_node, in_channels_edge, hidden_channels, aggr = "add"))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(tmp_in_channels_node, in_channels_edge, hidden_channels))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(tmp_in_channels_node, in_channels_edge, hidden_channels))
            else:
                raise ValueError("Undefined GNN type called.")

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, edge_attr=None):
        x = torch.cat([x_i, x_j], dim=1) if edge_attr is None else torch.cat([x_i, x_j, edge_attr], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
