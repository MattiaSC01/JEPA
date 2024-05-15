import os
import torch
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader

from src.jepa.utils import set_seed
from src.jepa.dataset import load_mnist, load_cifar, load_2Dtoy
from src.jepa.model.jepa import Jepa, JepaCriterion

from src.jepa.trainer.jepa_trainer import JepaTrainer
from src.jepa.sam import SAM

os.environ["WANDB__SERVICE_WAIT"] = "300"

# fixed hyperparams
load_dataset = load_2Dtoy
toy_dataset_type = 'circle'
toy_noise_scale = 0.0
in_dim = 2
hidden_dim = in_dim*10
train_size = 16384
test_size = 4096
batch_size = 64
lr = 0.0005
weight_decay = 5e-5
sparsity_weight = 0.0
max_epochs = 100
gpu_num = 2
device = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_num}"
compile_model = True
base_optimizer = torch.optim.SGD
base_optimizer = torch.optim.AdamW
optimizer_class = "ema"
# optimizer_class = "sam"
rho = 100.0
seed = 42
target_loss = 0.00
alpha = 0.985
log_to_wandb = True
log_images = True
log_interval = 10 # batches
checkpoint_interval = max_epochs # epochs
classification_interval = 10
classification_epochs = 3
wandb_project = "jepa-prove"

set_seed(seed)
print("Using device: ", device)

# dataset
root = "../data"
if load_dataset.__name__ == 'load_2Dtoy':
    train_dataset, test_dataset, train_metadata, test_metadata = load_dataset(
        dataset_type=toy_dataset_type, 
        noise_scale=toy_noise_scale,
        seed=seed
    )
else:
    train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, project=wandb_project, root=root, jepa=True, shuffle=seed, num_samples=train_size)
    test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, project=wandb_project, root=root, jepa=True, shuffle=seed, num_samples=test_size)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # be mindful of the batch size


# model components
encoder = torch.nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, in_dim),
    # nn.ReLU(),
    # nn.Linear(hidden_dim, in_dim),
)
predictor = torch.nn.Sequential(
    nn.Linear(in_dim, in_dim)
)

# init model and optim accordingly to the experiment
model = Jepa(encoder=encoder, predictor=predictor, seed=seed)
if optimizer_class.lower() == "sam":
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay, rho=rho)
else:
    optimizer = base_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = JepaCriterion(re=nn.MSELoss(), sparsity_weight=sparsity_weight)
scheduler = None


now = datetime.now()
wandb_run_name = f'{optimizer_class}-{base_optimizer.__name__}-{toy_dataset_type}--{now}'
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
    "checkpoint_root_dir": "checkpoints/",
    "target_loss": target_loss,
    "seed": seed,
    "compile_model": compile_model,
    "wandb_project": wandb_project,
    "wandb_run_name": wandb_run_name,
    "classification_interval": classification_interval,
    "classification_epochs": classification_epochs,
}

trainer = JepaTrainer(alpha=alpha, **train_config)
trainer.train()
