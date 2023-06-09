import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, GConvLSTM, TGCN, STGCN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class DCRNN_model(torch.nn.Module):
    def __init__(self, node_features):
        super(DCRNN_model, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


class DCRNN1(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(DCRNN1, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(input_size=64, hidden_size=64)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(64, num_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2) # shape: (batch_size, num_timesteps_input, num_nodes, num_features)
        x = x.reshape(batch_size * self.num_timesteps_input, self.num_nodes, self.num_features, 1) # shape: (batch_size * num_timesteps_input, num_nodes, num_features, 1)
        x = self.encoder(x) # shape: (batch_size * num_timesteps_input, 64, num_nodes, 1)
        x = x.reshape(batch_size, self.num_timesteps_input, 64, self.num_nodes) # shape: (batch_size, num_timesteps_input, 64, num_nodes)
        x = x.permute(1, 0, 3, 2) # shape: (num_timesteps_input, batch_size, num_nodes, 64)
        _, x = self.gru(x) # shape: (1, batch_size, 64)
        x = x.repeat(self.num_timesteps_output, 1, 1) # shape: (num_timesteps_output, batch_size, 64)
        x = x.permute(1, 2, 0).unsqueeze(-1) # shape: (batch_size, 64, num_timesteps_output, 1)
        x = self.decoder(x) # shape: (batch_size, num_features, num_timesteps_output, num_nodes)
        x = x.permute(0, 2, 3, 1) # shape: (batch_size, num_timesteps_output, num_nodes, num_features)
        return
