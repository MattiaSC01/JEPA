import torch
from torch import nn
from torch.utils.data import DataLoader
from jepa.utils import set_seed
from jepa.dataset import load_cifar, load_mnist
from jepa.model.autoencoder import AutoEncoder, AutoencoderCriterion
from jepa.trainer.ae_trainer import AutoencoderTrainer
from jepa.sam import SAM

# shallow: bs 64, lr 0.001, wd 0.1, rho 0.15
# deep: bs 64, wd 0.1, lr 0.0005, rho 0.1


# fixed hyperparams
load_dataset = load_cifar
N = 3072
B = 50
hidden_layers = [B]  # from first hidden to bottleneck, included
train_size = 16384
test_size = 4096
batch_size = 64
lr = 0.0005
weight_decay = 0.1
sparsity_weight = 0.0
max_epochs = 50
device = "cpu" if not torch.cuda.is_available() else "cuda"
compile_model = True
base_optimizer = torch.optim.AdamW
optimizer_class = "adamw"
rho = 0.05
seed = 72
target_loss = 0.00
print("Using device: ", device)


# logging
log_to_wandb = True
log_images = True
log_interval = 10 # batches
checkpoint_interval = max_epochs # epochs
flatness_interval = None # epochs
flatness_iters = 10
denoising_iters = 3
classification_interval = 10
classification_epochs = 3
wandb_project = "jepa-prove"


# dataset
root = "../data"
train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, project=wandb_project, root=root, jepa=False, shuffle=seed, num_samples=train_size)
test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, project=wandb_project, root=root, jepa=False, shuffle=seed, num_samples=test_size)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_size)  # be mindful of the batch size


# model
model = AutoEncoder(input_dim=N, encoder_hidden=hidden_layers, activation="ReLU", seed=seed)
if optimizer_class.lower() == "sam":
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay, rho=rho)
else:
    optimizer = base_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = AutoencoderCriterion(re=nn.MSELoss(), sparsity_weight=sparsity_weight)
scheduler = None


train_config = {
    "model": model,
    "optimizer": optimizer,
    "criterion": criterion,    
    "train_loader": train_loader,
    "test_loader": test_loader,
    "train_metadata": train_metadata,
    "test_metadata": test_metadata,
    "max_epochs": max_epochs,
    "device": device,
    "scheduler": scheduler,
    "log_to_wandb": log_to_wandb,
    "log_interval": log_interval,
    "log_images": log_images,
    "checkpoint_interval": checkpoint_interval,
    "checkpoint_root_dir": "../checkpoints",
    "flatness_interval": flatness_interval,
    "train_set_percentage_for_flatness": 'auto',
    "flatness_iters": flatness_iters,
    "denoising_iters": denoising_iters,
    "target_loss": target_loss,
    "seed": seed,
    "compile_model": compile_model,
    "wandb_project": wandb_project,
    "classification_interval": classification_interval,
    "classification_epochs": classification_epochs,
}

trainer = AutoencoderTrainer(**train_config)
trainer.train()
