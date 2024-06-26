import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jepa.trainer.ae_trainer import AutoencoderTrainer
from jepa.dataset import load_cifar, load_mnist
from jepa.model.autoencoder import AutoEncoder, AutoencoderCriterion
from jepa.sam import SAM
from jepa.utils import set_seed
from jepa.constants import PROJECT
import wandb
import yaml
import argparse


def get_dataset(
    config=None,
    dataset="cifar10",
):
    root = "../data"
    load_dataset = load_cifar if dataset == "cifar10" else load_mnist
    train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, root=root, jepa=False, shuffle=config.seed, num_samples=config.train_size)
    test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, root=root, jepa=False, shuffle=config.seed, num_samples=config.test_size)
    train_metadata["use_as"] = "train"
    test_metadata["use_as"] = "test"
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.test_size)  # be mindful of the batch size
    return train_loader, test_loader, train_metadata, test_metadata


def get_optimizer(config, model):
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "sam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=config.weight_decay, rho=config.rho)
    scheduler = None
    return optimizer, scheduler


def get_criterion(config):
    return AutoencoderCriterion(re=nn.MSELoss(), sparsity_weight=config.sparsity_weight)


def train(config=None):
    """
    Pass this function to wandb.agent to train the model using
    sweep hyperparameters.
    :param config: this will be overridden by the sweep config.
    You can use it to pass default values.
    """
    with wandb.init(config=config):
        config = wandb.config # this, apparently, will have been set by wandb.agent
        train_loader, test_loader, train_metadata, test_metadata = get_dataset(config, dataset=config.dataset.lower())
        model = AutoEncoder(input_dim=config.N, encoder_hidden=config.hidden_layers, activation=config.activation, seed=config.seed)
        optimizer, scheduler = get_optimizer(config, model)
        criterion = get_criterion(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        train_config = {
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,    
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_metadata": train_metadata,
            "test_metadata": test_metadata,
            "max_epochs": config.max_epochs,
            "device": device,
            "scheduler": scheduler,
            "log_to_wandb": True,
            "log_interval": config.log_interval,
            "log_images": config.log_images,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_root_dir": "../checkpoints",
            "flatness_interval": config.flatness_interval,
            "train_set_percentage_for_flatness": 'auto',
            "flatness_iters": config.flatness_iters,
            "denoising_iters": config.denoising_iters,
            "target_loss": config.target_loss,
            "seed": config.seed,
            "compile_model": config.compile_model,
            "is_sweep": True,
            "classification_interval": config.classification_interval,
            "classification_epochs": config.classification_epochs,
        }
        
        trainer = AutoencoderTrainer(**train_config)
        trainer.train()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='sweep.yaml', help='Path to the config file')
parser.add_argument('--project', type=str, default=PROJECT, help='Wandb project name')
parser.add_argument('--count', type=int, default=1, help='Number of runs')
args = parser.parse_args()

with open(args.config, "r") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)
sweep_id = wandb.sweep(sweep_config, project=args.project)
agent = wandb.agent(sweep_id, function=train, count=args.count)
