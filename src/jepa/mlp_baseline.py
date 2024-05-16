import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from src.jepa.utils import set_seed
from src.jepa.dataset import load_2Dtoy

# fixed hyperparams
toy_dataset_type = 'circle'
toy_noise_scale = 0.0
in_dim = 2
hidden_dim = in_dim*10
batch_size = 64
lr = 0.001
weight_decay = 0.005
max_epochs = 100
gpu_idx = 2
device = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_idx}"
seed = 420
set_seed(seed)

# This architecture corresponds to the encoder mlps we use for the jepa model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(self.fc2(x))

        return x

# dataset
root = "../data"
train_dataset, test_dataset, train_metadata, test_metadata = load_2Dtoy(
    dataset_type=toy_dataset_type, 
    noise_scale=toy_noise_scale,
    seed=seed
)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Data shapes')
print(len(train_dataset), len(test_dataset))

# Fit models
model = MLP(in_dim, hidden_dim, 1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()
print(f'Training on {toy_dataset_type} dataset on device {device}...')
for epoch in range(max_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch in train_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        optimizer.zero_grad()
        outputs = model(x).flatten()
        loss = criterion(outputs, y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = len(test_dataset)
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            outputs = model(x).flatten()
            preds = torch.round(torch.sigmoid(outputs))
            total_correct += (y == preds).sum()

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}")

# Visualize supervised MLP representations
model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = len(test_dataset)
zs, ys = [], []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y']
        z = model.fc2(F.relu(model.fc1(x)))
        zs.append(z.detach().cpu().numpy())
        ys.append(y.numpy())

# make a scatterplot of zs with the colors indicating the label given in ys
zs, ys = np.vstack(zs), np.hstack(ys)
plt.scatter(zs[:, 0], zs[:, 1], c=ys)
plt.savefig(f'mlp-repr-{toy_dataset_type}.png')
plt.close()
print('Training completed!')