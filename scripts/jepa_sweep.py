import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jepa.trainer.jepa_trainer import JepaTrainer
from jepa.dataset import load_cifar, load_mnist
from jepa.model.jepa import Jepa, JepaCriterion
from jepa.sam import SAM
from jepa.utils import set_seed, sequential_from_string
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
    train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, root=root, jepa=True, shuffle=config.seed, num_samples=config.train_size)
    test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, root=root, jepa=True, shuffle=config.seed, num_samples=config.test_size)
    train_metadata["use_as"] = "train"
    test_metadata["use_as"] = "test"
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.test_size)  # be mindful of the batch size
    return train_loader, test_loader, train_metadata, test_metadata


def get_model(config):
    encoder = sequential_from_string(config.encoder)
    if config.predictor == "auto":
        n_layers = len(encoder) - 1
        # fails if the last layer is not a Linear layer!
        while not isinstance(encoder[n_layers], nn.Linear):
            n_layers -= 1
        B = encoder[n_layers].out_features
        predictor = nn.Sequential(nn.Linear(B, B))
    else:
        predictor = sequential_from_string(config.predictor)
    return Jepa(encoder=encoder, predictor=predictor, seed=config.seed)


def get_optimizer(config, model):
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "sam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=config.lr, weight_decay=config.weight_decay, rho=config.rho)
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    scheduler = None
    return optimizer, scheduler


def get_criterion(config):
    return JepaCriterion(re=nn.MSELoss(), sparsity_weight=config.sparsity_weight)


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
        model = get_model(config)
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
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint_root_dir": "../checkpoints",
            "target_loss": config.target_loss,
            "seed": config.seed,
            "compile_model": config.compile_model,
            "is_sweep": True,
            "classification_interval": config.classification_interval,
            "classification_epochs": config.classification_epochs,
            "alpha": config.alpha,
        }
        
        trainer = JepaTrainer(**train_config)
        trainer.train()


# # debug (also comment `config = wandb.config` in train function)
# import yaml
# with open("/home/3144860/JEPA/config/jepa_sweep.yaml", "r") as f:
#     sweep_config = yaml.load(f, Loader=yaml.FullLoader)
# class DummyConfig:
#     def __init__(self, config):
#         for key, value in config.items():
#             if "value" in value:
#                 value = value["value"]
#             else:
#                 value = value["values"]
#             setattr(self, key, value)
# dummy_config = DummyConfig(sweep_config['parameters'])
# train(dummy_config)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='sweep.yaml', help='Path to the config file')
parser.add_argument('--project', type=str, default=PROJECT, help='Wandb project name')
parser.add_argument('--count', type=int, default=1, help='Number of runs')
args = parser.parse_args()

with open(args.config, "r") as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)
sweep_id = wandb.sweep(sweep_config, project=args.project)
agent = wandb.agent(sweep_id, function=train, count=args.count)
