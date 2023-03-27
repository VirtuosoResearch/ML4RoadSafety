import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, GConvLSTM, TGCN, STGCN


class STGCN_model(torch.nn.Module):
    def __init__(self, node_features):
        super(STGCN_model, self).__init__()
        self.recurrent = STGCN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h