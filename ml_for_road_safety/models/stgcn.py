from functools import reduce
from operator import mul

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from layers import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, int((kernel_size-1)/2)))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, int((kernel_size-1)/2)))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, int((kernel_size-1)/2)))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConv(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(
        self,
        num_nodes: int,
        in_channels_node: int,
        in_channels_edge: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        K: int = 1,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels_node,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = GCNConv(in_channels_node=hidden_channels,
                        in_channels_edge=in_channels_edge,
                        out_channels=hidden_channels)

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        # self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)
        # T = torch.zeros_like(T_0).to(T_0.device)
        # for b in range(T_0.size(0)):
        #     for t in range(T_0.size(1)):
        #         T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)
        timesteps = T_0.size(-3); batch_size = T_0.size(0)
        T_0 = T_0.reshape(reduce(mul, T_0.size()[:-2]), *T_0.size()[-2:])
        data = self.__batch_timesteps__(T_0, edge_index,
                                            edge_weight).to(T_0.device)
        
        T = self._graph_conv(data.x, data.edge_index, data.edge_attr if data.edge_attr is not None else None)
        T = T.reshape(batch_size, timesteps, -1, self._graph_conv.out_channels)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        # T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        return T

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
    
class STGCN(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers, dropout = 0, JK = "last", num_nodes=-1):
        super(STGCN, self).__init__()
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
            
            self.gnns.append(STConv(num_nodes, tmp_in_channels_node, in_channels_edge, hidden_channels, hidden_channels))
            

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