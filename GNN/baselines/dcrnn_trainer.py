from dcrnn import *

from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader

dataloader = METRLADatasetLoader()
# dataloader = PemsBayDatasetLoader()

train_dataloader, test_dataloader = dataloader.get_dataloaders(batch_size=32), dataloader.get_dataloaders(batch_size=32)

model = DCRNN(num_nodes=207, num_features=2, num_timesteps_input=12, num_timesteps_output=12)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

num_epochs = 200

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
        
        
# from tqdm import tqdm

# model = RecurrentGCN(node_features = 4)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# model.train()

# for epoch in tqdm(range(200)):
#     cost = 0
#     # for regression, change to classification when needed
#     for time, snapshot in enumerate(train_dataset):
#         y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#         cost = cost + torch.mean((y_hat-snapshot.y)**2)
#     cost = cost / (time+1)
#     cost.backward()
#     optimizer.step()
#     optimizer.zero_grad()