import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Optional, Union
import os
import json
from .evaluation import EvalAE
from .trainer import Trainer
from .utils import set_seed


class AutoEncoder(nn.Module):
    """
    Autoencoder made of a sequence of fully connected layers.
    Architecture is symmetric: encoder and decoder have the same
    linear layers, but in reverse order.
    """
    def __init__(
            self,
            input_dim: int,
            encoder_hidden: list[int],
            activation: Optional[str] = None,
            seed: int = 42,
            decoder_hidden: Optional[list] = None,
        ):
        """
        :param input_dim: dimension of the input data
        :param hidden_layers: list of dimensions for the hidden layers.
        first is the first hidden layer, last is the bottleneck.
        :param activation: activation function to use. If None, defaults to "ReLU".
        :param seed: random seed for weight initialisation.
        :param decoder_hidden_layers: if None, build decoder symmetrically wrt encoder.
        Otherwise, expects a list of dimensions for the hidden layers. Bottleneck and
        input layers are both excluded (contrary to encoder!).
        """
        super().__init__()
        if activation is None:
            activation = "ReLU"
        if decoder_hidden is None:
            decoder_hidden = list(reversed(encoder_hidden[:-1]))
        self.input_dim = input_dim
        self.bottleneck = encoder_hidden[-1]
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.activation = activation
        self.seed = seed
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.initialize_weights(self.seed)
    
    @classmethod
    def from_pretrained(cls, chkpt_dir: str, device: str = "cpu") -> tuple["AutoEncoder", dict]:
        metadata_path = os.path.join(chkpt_dir, "metadata.json")
        weights_path = os.path.join(chkpt_dir, "weights.pt")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        architecture = metadata["architecture"]
        input_dim = architecture["input_dim"]
        hidden_layers = architecture["hidden_layers"]
        activation = architecture["activation"]
        decoder_hidden = architecture["decoder_hidden"]
        ae = cls(input_dim, hidden_layers, activation, decoder_hidden=decoder_hidden)
        ae.load_state_dict(torch.load(weights_path, map_location=device))
        return ae, metadata

    def build_encoder(self):
        layers = []
        prev_layer = self.input_dim
        for layer_dim in self.encoder_hidden:
            layers.append(nn.Linear(prev_layer, layer_dim))
            layers.append(getattr(nn, self.activation)())
            prev_layer = layer_dim
        layers = layers[:-1]  # no activation before bottleneck
        return nn.Sequential(*layers)
    
    def build_decoder(self):
        layers = []
        prev_layer = self.bottleneck
        for layer_dim in self.decoder_hidden:
            layers.append(nn.Linear(prev_layer, layer_dim))
            layers.append(getattr(nn, self.activation)())
            prev_layer = layer_dim
        layers.append(nn.Linear(prev_layer, self.input_dim))
        return nn.Sequential(*layers)
    
    def initialize_weights(self, seed):
        set_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {"x_hat": x_hat, "z": z}
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def compute_parameter_norm(self):
        """
        Compute the l2 norm of weights and biases
        of the linear layers.
        :return: tuple of norms (weight_norm, bias_norm), 
        normalized by the number of parameters.
        """
        weight_norm, bias_norm = 0.0, 0.0
        weight_count, bias_count = 0, 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_norm += m.weight.norm(2)
                bias_norm += m.bias.norm(2)
                weight_count += m.weight.numel()
                bias_count += m.bias.numel()
        return weight_norm / weight_count, bias_norm / bias_count
    
    def get_architecture(self) -> dict:
        """
        Return a dictionary encoding the model's architecture, for logging and
        to easily rebuild the model.
        """
        architecture = {
            "input_dim": self.input_dim,
            "hidden_layers": self.encoder_hidden,
            "activation": self.activation,
            "decoder_hidden": self.decoder_hidden,
        }
        return architecture
    

class AutoencoderTrainer(Trainer):
    def __init__(
            self,
            flatness_interval: Optional[int] = None,
            train_set_percentage_for_flatness: Union[float, str] = 1.0,
            flatness_iters: int = 5,
            denoising_iters: int = 1,
            **kwargs
        ):
        """
        Trainer specific to autoencoders. Adds flatness and denoising evaluation.
        :param flatness_interval: compute flatness every flatness_interval epochs. If None,
        flatness is not computed. To compute only at the end of training, set it to epochs.
        :param train_set_percentage_for_flatness: percentage of the training set to use for
        computing flatness. If "auto", use as many samples as there are in the test set.
        :param flatness_iters: number of repetitions to compute flatness. Higher values
        give more accurate results, but take longer to compute.
        :param denoising_iters: same as flatness_iters, but for denoising.
        """
        super().__init__(**kwargs)
        if flatness_interval is None:
            flatness_interval = self.max_epochs + 1  # no flatness
        if train_set_percentage_for_flatness == 'auto':
            assert self.test_loader is not None, "If train_set_percentage_for_flatness is 'auto', test_loader must be provided."
            train_set_percentage_for_flatness = min(len(self.test_loader.dataset) / len(self.train_loader.dataset), 1.0)
        self.flatness_interval = flatness_interval
        self.train_set_percentage_for_flatness = train_set_percentage_for_flatness
        self.flatness_iters = flatness_iters
        self.denoising_iters = denoising_iters
    
    @torch.no_grad()
    def test_epoch(self):
        loss = super().test_epoch()
        if self.epoch % self.flatness_interval == 0:
            self.handle_flatness()
            self.handle_denoising()
        if self.log_to_wandb and self.log_images:
            self.log_images_to_wandb(split="val")
            self.log_images_to_wandb(split="train")
        return loss
    
    def end_training(self):
        if self.checkpoint_interval <= self.max_epochs and self.epoch % self.checkpoint_interval != 0:
            self.make_checkpoint()
        if self.flatness_interval <= self.max_epochs and self.epoch % self.flatness_interval != 0:
            self.handle_flatness()
            self.handle_denoising()
        if self.log_to_wandb:
            self.logger.end_run()
        
    def handle_flatness(self):
        sigmas = np.linspace(0, 0.15, 10)
        n_iters = self.flatness_iters
        flatness_train = EvalAE.flatness_profile(self.model, self.train_loader, sigmas, n_iters, criterion=self.criterion, data_percentage=self.train_set_percentage_for_flatness)
        self.log_flatness(flatness_train, split="train")
        flatness_val = EvalAE.flatness_profile(self.model, self.test_loader, sigmas, n_iters, criterion=self.criterion)
        self.log_flatness(flatness_val, split="val")

    def handle_denoising(self):
        sigmas = np.linspace(0, 1.0, 10)
        n_iters = self.denoising_iters
        denoising_train = EvalAE.denoising_profile(self.model, self.train_loader, sigmas, n_iters, criterion=self.criterion, data_percentage=self.train_set_percentage_for_flatness)
        self.log_denoising(denoising_train, split="train")
        denoising_val = EvalAE.denoising_profile(self.model, self.test_loader, sigmas, n_iters, criterion=self.criterion)
        self.log_denoising(denoising_val, split="val")

    def log_flatness(self, losses: dict, split: str):
        """
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        """
        df = pd.DataFrame(losses)
        df.columns = [f"{sigma:.3f}" for sigma in df.columns]
        for sigma in df.columns:
            mean = df[sigma].mean()
            std = df[sigma].std()
            self.logger.log_metric(mean, f"flatness/{split}/mean_{sigma}", self.step)
            self.logger.log_metric(std, f"flatness/{split}/std_{sigma}", self.step)
        avg_diff = df.mean().mean() - df.iloc[0, 0]
        self.logger.log_metric(avg_diff, f"flatness/{split}/avg_diff", self.step)
        self.logger.log_metric(avg_diff, f"{split}/flatness_avg_diff", self.step) # redundant, but useful in the dashboard.
        # self.logger.log_table(df, f"tables/{split}_flatness", self.step)
        self.plot_and_log(losses, split=split, plot_type="flatness")

    def log_denoising(self, losses: dict, split: str):
        """
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        """
        df = pd.DataFrame(losses)
        df.columns = [f"{sigma:.3f}" for sigma in df.columns]
        for sigma in df.columns:
            mean = df[sigma].mean()
            std = df[sigma].std()
            self.logger.log_metric(mean, f"denoising/{split}/mean_{sigma}", self.step)
            self.logger.log_metric(std, f"denoising/{split}/std_{sigma}", self.step)
        avg_diff = df.mean().mean() - df.iloc[0, 0]
        self.logger.log_metric(avg_diff, f"denoising/{split}/avg_diff", self.step)
        self.logger.log_metric(avg_diff, f"{split}/denoising_avg_diff", self.step) # redundant, but useful in the dashboard.
        # self.logger.log_table(df, f"tables/{split}_denoising", self.step)
        self.plot_and_log(losses, split=split, plot_type="denoising")

    @torch.no_grad()
    def log_images_to_wandb(self, split: str = "val", n_images: int = 1):
        """
        Log some pairs of input and output images to wandb.
        For now, only works with MNIST-like images.
        :param split: "val" or "train"
        :param n_images: number of distinct images to try
        """
        ds = self.test_loader.dataset if split == "val" else self.train_loader.dataset
        strengths = [0.0, 0.0, 0.5, 1.0, 0.25, 0.5, 0.25, 0.5]
        types = [
            "identity",
            "identity",
            "gaussian-additive",
            "gaussian-additive",
            "salt-and-pepper",
            "salt-and-pepper",
            "dropout",
            "dropout",
        ]
        images = []
        for _ in range(n_images):
            idx = random.randint(0, len(ds) - 1)
            for strength, noise_type in zip(strengths, types):
                x = ds[idx]["x"].to(self.device)
                noisy = EvalAE.corrupt_data(x, strength, noise_type=noise_type)
                x_hat = self.model(noisy)["x_hat"]
                noisy = self.reassemble_image(noisy)
                x_hat = self.reassemble_image(x_hat)
                images.extend([noisy, x_hat])
        self.logger.log_tensor_as_image(images, f"{split}/images", self.step)

    def reassemble_image(self, x):
        """
        :param x: a single image tensor (C*H*W,)
        :return: a reshaped image tensor (C, H, W). dimensions
        are inferred from the dataset metadata.
        """
        name = self.train_metadata["id"]
        if "cifar" in name.lower():
            x = x.reshape(32, 32, 3)
            x = x.permute(2, 0, 1)
        elif "mnist" in name.lower():
            x = x.reshape(1, 28, 28)
        else:
            raise ValueError(f"Dataset {name} not recognized.")
        return x
    
    def plot_and_log(self, losses: dict, split: str, plot_type: str):
        """
        If run is within a sweep, do nothing (cannot use matplotlib GUI from a non-main thread).
        :param losses: dictionary with noise_strengths as keys and lists of losses as values
        :param split: "train" or "val"
        :param plot_type: "flatness" or "denoising"
        """
        if self.is_sweep:
            return
        plt.figure(figsize=(10, 5))
        plt.title(f"{plot_type} profile")
        EvalAE.plot_profile(losses, color="blue", label=split)
        plt.legend()
        self.logger.log_plot(f"{split}/{plot_type}", self.step)
        plt.close()


class AutoencoderCriterion(nn.Module):
    def __init__(
            self,
            re: Optional[nn.Module] = None,
            sparsity_weight: float = 0.0,
        ):
        """
        :param re: reconstruction error criterion (e.g. nn.MSELoss)
        """
        super().__init__()
        if re is None:
            re = nn.MSELoss(reduction="mean")
        self.re = re
        self.sparsity_weight = sparsity_weight
    
    def forward(self, output: dict, batch: dict) -> dict:
        """
        :param output: dict with keys "x_hat", "z"
        :param batch: dict with key "x"
        :return: dict with keys "loss", "re", "latent_l1norm"
        """
        x = batch["x"]
        x_hat, z = output["x_hat"], output["z"]
        re = nn.functional.mse_loss(x_hat, x)
        latent_l1norm = torch.norm(z, p=1).mean()
        loss = re + self.sparsity_weight * latent_l1norm
        return {"loss": loss, "re": re, "latent_l1norm": latent_l1norm}

    def get_config(self) -> dict:
        return {
            "criterion": type(self).__name__,
            "reconstruction_error": type(self.re).__name__,
            "sparsity_weight": self.sparsity_weight
        }
