# class DCRNN():
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.build_model()
#         self.model.to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         self.loss = torch.nn.MSELoss()
        
import torch
import torch.nn as nn
# https://blog.csdn.net/m0_53961910/article/details/128135170?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-128135170-blog-127385319.pc_relevant_recovery_v2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
class DCRNN_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(DCRNN, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define the CNN layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Define the RNN layers
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Pass the input through the CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Pass the output through the RNN layers
        x, _ = self.rnn(x)
        
        # Pass the output through the output layer
        x = self.fc(x[:, -1, :])
        
        return x

import torch
import torch.nn as nn

class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DCRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Define the CNN layers
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Define the RNN layers
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the CNN layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Pass the output of the CNN layers through the RNN layers
        x, _ = self.rnn(x)

        # Pass the output of the RNN layers through the output layer
        x = self.fc(x[:, -1, :])

        return x

# Define the model, loss function, and optimizer
model = DCRNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(num_epochs):
    # Loop over the training data
    for inputs, targets in train_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

# Test the model
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
