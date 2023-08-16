import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from layers import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class AVWGCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int
    ):
        super(AVWGCN, self).__init__()
        self.K = K
        self.weights_pool = torch.nn.Parameter(
            torch.Tensor(embedding_dimensions, K, in_channels, out_channels)
        )
        self.bias_pool = torch.nn.Parameter(
            torch.Tensor(embedding_dimensions, out_channels)
        )
        glorot(self.weights_pool)
        zeros(self.bias_pool)

    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """

        number_of_nodes = E.shape[0]
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(number_of_nodes).to(supports.device), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum("nd,dkio->nkio", E, self.weights_pool)
        bias = torch.matmul(E, self.bias_pool)
        X_G = torch.einsum("knm,bmc->bknc", supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum("bnki,nkio->bno", X_G, W) + bias
        return X_G


class AGCRN(nn.Module):
    r"""An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(
        self,
        number_of_nodes: int,
        in_channels: int,
        in_channels_edge: int,
        out_channels: int,
        K: int = 3,
        embedding_dimensions: int = 128,
    ):
        super(AGCRN, self).__init__()

        self.number_of_nodes = number_of_nodes
        self.in_channels = in_channels
        self.in_channels_edge = in_channels_edge
        self.out_channels = out_channels
        self.K = K
        self.embedding_dimensions = embedding_dimensions
        self._setup_layers()

    def _setup_layers(self):
        self._gate = GCNConv(in_channels_node=self.in_channels + self.out_channels,
                             in_channels_edge=self.in_channels_edge,
                             out_channels=2 * self.out_channels,)
        # AVWGCN(
        #     in_channels=self.in_channels + self.out_channels,
        #     out_channels=2 * self.out_channels,
        #     K=self.K,
        #     embedding_dimensions=self.embedding_dimensions,
        # )

        self._update = GCNConv(in_channels_node=self.in_channels + self.out_channels,
                             in_channels_edge=self.in_channels_edge,
                             out_channels=self.out_channels,)
        # AVWGCN(
        #     in_channels=self.in_channels + self.out_channels,
        #     out_channels=self.out_channels,
        #     K=self.K,
        #     embedding_dimensions=self.embedding_dimensions,
        # )

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def forward(
        self, X: torch.FloatTensor, edge_index, edge_attr = None, H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix.
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        X = X.squeeze(0) # input shape: (B, T, N, F) -> (T, N, F) since B = 1
        H = self._set_hidden_state(X, H)
        X_H = torch.cat((X, H), dim=-1)

        timesteps = X.shape[0]
        data = self.__batch_timesteps__(X_H, edge_index,
                                            edge_attr).to(X_H.device)
        Z_R = torch.sigmoid(self._gate(data.x, data.edge_index, data.edge_attr))
        Z_R = torch.reshape(Z_R, (timesteps, -1, self.out_channels * 2))

        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X, Z * H), dim=-1)

        data = self.__batch_timesteps__(C, edge_index,
                                            edge_attr).to(C.device)
        HC = torch.tanh(self._update(data.x, data.edge_index, data.edge_attr))
        HC = torch.reshape(HC, (timesteps, -1, self.out_channels))
        H = R * H + (1 - R) * HC
        return H
    
    def __batch_timesteps__(self, x, edge_index, edge_weight=None):
        r"""
        This method batches the data for several time steps into a single batch
        """
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)

        if edge_weight is not None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)

        if edge_weight is not None:
            dataset = [
                Data(x=_x, edge_index=e_i, edge_attr=e_w)
                for _x, e_i, e_w in zip(x, edge_index, edge_weight)
            ]
        else:
            dataset = [
                Data(x=_x, edge_index=e_i)
                for _x, e_i in zip(x, edge_index)
            ]
        loader = DataLoader(dataset=dataset, batch_size=x.size(0))

        return next(iter(loader))

class AGCRN_Model(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers, dropout = 0, JK = "last", num_nodes=-1):
        super(AGCRN_Model, self).__init__()
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
            
            self.gnns.append(AGCRN(num_nodes, tmp_in_channels_node, in_channels_edge, hidden_channels))
            

    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
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

                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    x = F.dropout(x, self.drop_ratio, training = self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training = self.training)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
