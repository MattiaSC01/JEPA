import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.jepa.utils import set_seed
from src.jepa.dataset import load_2Dtoy, load_cifar, load_mnist



# fixed hyperparams
# load_dataset = load_cifar
load_dataset = load_mnist
# load_dataset = load_2Dtoy
toy_dataset_type = 'circle'
toy_noise_scale = 0.0
batch_size = 64
lr = 0.0001
weight_decay = 0.005
max_epochs = 100
gpu_idx = 1
device = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_idx}"
seed = 42
set_seed(seed)

# This architecture corresponds to the encoder mlps we use for the jepa model + readout
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dataset='toy'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if 'toy' in dataset:
            self.fc2 = nn.Linear(hidden_size, input_size)
            self.fc3 = nn.Linear(input_size, output_size)
        else:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(self.fc2(x))

        return x
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# dataset
root = "../data"
if load_dataset.__name__ == 'load_2Dtoy':
    train_dataset, test_dataset, train_metadata, test_metadata = load_dataset(
        dataset_type=toy_dataset_type, 
        noise_scale=toy_noise_scale,
        seed=seed
    )
else:
    train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, root=root, jepa=True, shuffle=seed)
    test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, root=root, jepa=True, shuffle=seed)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Data shapes')
print(len(train_dataset), len(test_dataset))

# Fit models
if load_dataset.__name__ == 'load_2Dtoy':
    in_dim = 2
    hidden_dim = in_dim*10
    out_dim = 10
else:
    in_dim = train_dataset[0]['x'].flatten().shape[0]
    if load_dataset.__name__ == 'load_cifar':
        hidden_dim = 1024
    else:
        hidden_dim = 256
    out_dim = 10

model = MLP(in_dim, hidden_dim, out_dim, dataset=load_dataset.__name__ ).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss() if load_dataset.__name__ == 'load_2Dtoy' else nn.CrossEntropyLoss()
print(f'Training on device {device}...')
for epoch in range(max_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch in train_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        optimizer.zero_grad()
        if load_dataset.__name__ == 'load_2Dtoy':
            outputs = model(x).flatten()
            loss = criterion(outputs, y.float())
        else:
            outputs = model(x)
            loss = criterion(outputs, y)

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
            if load_dataset.__name__ == 'load_2Dtoy':
                outputs = model(x).flatten()
                preds = torch.round(torch.sigmoid(outputs))
            else:
                outputs = model(x)
                preds = outputs.argmax(1)
            total_correct += (preds == y).sum()

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
        z = model.encode(x)
        zs.append(z.detach().cpu().numpy())
        ys.append(y.numpy())

# make a scatterplot of zs with the colors indicating the label given in ys
zs, ys = np.vstack(zs), np.hstack(ys)
if load_dataset.__name__ == 'load_2Dtoy':
    plt.scatter(zs[:, 0], zs[:, 1], c=ys)
    plt.savefig(f'mlp-repr-{toy_dataset_type}.png')
    plt.close()
else:
    zs_sne = TSNE(n_components=2, random_state=seed).fit_transform(zs)
    zs_pca = PCA(n_components=2, random_state=seed).fit_transform(zs)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('TSNE')
    plt.scatter(zs_sne[:, 0], zs_sne[:, 1], c=ys)
    plt.subplot(122)
    plt.title('PCA')
    plt.scatter(zs_pca[:, 0], zs_pca[:, 1], c=ys)
    plt.tight_layout()
    plt.savefig(f'mlp-repr-{load_dataset.__name__}.png')
    plt.close()

print('MLP training and evaluation completed!')
