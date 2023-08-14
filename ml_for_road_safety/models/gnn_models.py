import torch
import torch.nn.functional as F
from layers import GCNConv, GraphSAGEConv, GATConv, GINConv
from torch_geometric_temporal.nn import DCRNN, TGCN

class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_index, edge_attr):
        return x

class GNN(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers, dropout = 0, JK = "last", gnn_type = "gcn", num_nodes=-1):
        super(GNN, self).__init__()
        self.num_layer = num_layers
        self.drop_ratio = dropout
        self.JK = JK

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be greater than 0.")
        
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
            elif gnn_type == "dcrnn":
                self.gnns.append(DCRNN(tmp_in_channels_node, hidden_channels, K=1))
            elif gnn_type == "tgcn":
                self.gnns.append(TGCN(tmp_in_channels_node, hidden_channels))
            # elif gnn_type == "stgcn":
            #     self.gnns.append(STConv(num_nodes, tmp_in_channels_node, hidden_channels, hidden_channels, 1, 1))
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
    
    def forward_ns(self, x, adjs, edge_attr):
        h_list = [x]
        
        for layer, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            edge_attr_target = edge_attr[e_id]
            h = self.gnns[layer]((x, x_target), edge_index, edge_attr_target)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
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

    def inference(self, x_all, subgraph_loader, edge_attr, device):
        total_edges = 0
        for layer in range(self.num_layer):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_ids, size = adj.to(device)
                total_edges += edge_index.size(1)

                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                edge_attr_target = edge_attr[e_ids].to(device)
                x = self.gnns[layer]((x, x_target), edge_index, edge_attr_target)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    x = F.dropout(x, self.drop_ratio, training = self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training = self.training)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, if_regression = False):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.if_regression = if_regression

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
        if self.if_regression:
            return x
        return torch.sigmoid(x)
    
