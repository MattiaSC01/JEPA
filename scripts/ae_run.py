import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

from src.jepa.utils import set_seed
from src.jepa.dataset import load_cifar, load_mnist
from src.jepa.model.autoencoder import AutoEncoder, AutoencoderCriterion
from src.jepa.trainer.ae_trainer import AutoencoderTrainer
from src.jepa.sam import SAM

# shallow: bs 64, lr 0.001, wd 0.1, rho 0.15
# deep: bs 64, wd 0.1, lr 0.0005, rho 0.1


# fixed hyperparams
load_dataset = load_cifar
# load_dataset = load_mnist
batch_size = 128
lr = 0.0005
weight_decay = 0.5
sparsity_weight = 0.0
max_epochs = 50
gpu_idx = 2
device = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_idx}"
compile_model = True
base_optimizer = torch.optim.AdamW
optimizer_class = "adamw"
rho = 0.05
seed = 42
target_loss = 0.00

set_seed(seed)
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
train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, project=wandb_project, root=root, jepa=False, shuffle=seed)
test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, project=wandb_project, root=root, jepa=False, shuffle=seed)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  # be mindful of the batch size


# model
in_dim = train_dataset[0]['x'].shape[0]
if load_dataset.__name__ == 'load_cifar':
    hidden_dim = 1024
else:
    hidden_dim = 256
model = AutoEncoder(input_dim=in_dim, encoder_hidden=[hidden_dim, hidden_dim], activation="ReLU", seed=seed)
if optimizer_class.lower() == "sam":
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay, rho=rho)
else:
    optimizer = base_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = AutoencoderCriterion(re=nn.MSELoss(), sparsity_weight=sparsity_weight)
scheduler = None

now = datetime.now()
wandb_run_name = f'AE-{optimizer_class}-{load_dataset.__name__}--{now}'
    

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
    "wandb_run_name": wandb_run_name,
    "classification_interval": classification_interval,
    "classification_epochs": classification_epochs,
}

trainer = AutoencoderTrainer(**train_config)
trainer.train()
