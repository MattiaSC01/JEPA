import torch
from torch import nn
from typing import Optional, Union
import os
import json
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
