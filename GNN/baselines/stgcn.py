import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GRU

class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DCRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Define the CNN layers
        self.conv1 = GraphConv(input_size, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        
        # Define the RNN layers
        self.rnn = GRU(hidden_size, hidden_size, num_layers)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # Pass the input through the CNN layers
        x = self.conv1(data.x, data.edge_index)
        x = self.conv2(x, data.edge_index)

        # Pass the output of the CNN layers through the RNN layers
        x, _ = self.rnn(x)

        # Pass the output of the RNN layers through the output layer
        x = self.fc(x[:, -1, :])

        return x

# Define the model, loss function, and optimizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GraphConv, LSTM

class STGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(STGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Define the GCN layers
        self.gc1 = GraphConv(input_size, hidden_size)
        self.gc2 = GraphConv(hidden_size, hidden_size)
        
        # Define the TCN layers
        self.tcn = LSTM(hidden_size, hidden_size, num_layers)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # Pass the input through the GCN layers
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)

        # Pass the output of the GCN layers through the TCN layers
        x, _ = self.tcn(x)

        # Pass the output of the TCN layers through the output layer
        x = self.fc(x[:, -1, :])

        return x

# Load the training and test data
train_dataset = YourDataset(...)
test_dataset = YourDataset(...)

# Define the model, loss function, and optimizer
model = STGCN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for data in train_dataloader:
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
    for data in test_dataloader:
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        print('Test loss:', loss.item())

# Analyze the effect of temperature scaling on the model's calibration
# predicted_probabilities = predicted_probabilities ^ (1 / temperature)

