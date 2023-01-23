import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.datasets import METR
from torch_geometric.nn import GCNConv, GRU

# Load the METR-LA dataset
dataset = METR(root='/path/to/dataset')

# Define the model
class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DCRNN, self).__init__()

        # Define the GCN layers
        self.gc1 = GCNConv(input_size, hidden_size)
        self.gc2 = GCNConv(hidden_size, hidden_size)

        # Define the GRU layers
        self.gru = GRU(hidden_size, hidden_size)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # Pass the input through the GCN layers
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)

        # Pass the output of the GCN layers through the GRU layers
        x, _ = self.gru(x)

        # Pass the output of the GRU layers through the output layer
        x = self.fc(x[:, -1, :])

        return x

# Define the input size, hidden size, and output size
input_size = dataset.num_features
hidden_size = 128
output_size = 1

# Create the model
model = DCRNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for data in dataset:
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.datasets import METR
from torch_geometric.nn import GCNConv, GRU

# Load the METR-LA dataset
dataset = METR(root='/path/to/dataset')

# Define the model
class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DCRNN, self).__init__()

        # Define the GCN layers
        self.gc1 = GCNConv(input_size, hidden_size)
        self.gc2 = GCNConv(hidden_size, hidden_size)

        # Define the GRU layers
        self.gru = GRU(hidden_size, hidden_size)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # Pass the input through the GCN layers
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)

        # Pass the output of the GCN layers through the GRU layers
        x, _ = self.gru(x)

        # Pass the output of the GRU layers through the output layer
        x = self.fc(x[:, -1, :])

        return x

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for data in dataset:
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

# Test the model
with torch.no_grad():
    for data in dataset:
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        print('Test loss:', loss.item())
