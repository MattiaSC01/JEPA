from typing import Optional
import torch
from torch import nn
from copy import deepcopy
import os
import json
from .trainer import Trainer
from .utils import sequential_from_string, set_seed


# TODO: create methods to monitor relevant metrics 
#       (e.g. overlap encoder-ema, norms, etc.)
# TODO: jepa training is slow as hell rn. Probably due to copying the weights.
#       fix that. If still slow, profile and optimize.


class Jepa(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            predictor: nn.Module,
            seed: int = 42,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.initialize_weights(seed)
        self.ema = deepcopy(encoder)
        for param in self.ema.parameters():
            param.requires_grad = False

    def initialize_weights(self, seed: int):
        # TODO: consider more general architectures (for now, we fall back to default init except for Linear layers)
        set_seed(seed)  # unique seed at the beginning --> change in any module will affect all
        for module in [self.encoder, self.predictor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        encoder_output = self.encoder(x)
        predictor_output = self.predictor(encoder_output)
        ema_output = self.ema(x)
        return {"encoder_output": encoder_output, "predictor_output": predictor_output, "ema_output": ema_output}
    
    @torch.no_grad()
    def update_ema(self, alpha):
        """
        TODO: make this more efficient: avoid copying the weights, rather update
              them in-place considering momentum and lr of self.encoder. Might 
              require a custom optimizer.
        Update weights of the EMA encoder.
        :param alpha: momentum for ema update. If 0, just copy the weights.
        """
        for ema_param, param in zip(self.ema.parameters(), self.encoder.parameters()):
            ema_param.copy_((1 - alpha) * param + alpha * ema_param)

    def get_architecture(self) -> dict:
        """
        Return a dictionary encoding the model's architecture, for logging
        and to easily rebuild the model.
        """
        architecture = {
            "encoder": str(self.encoder),
            "predictor": str(self.predictor),
        }
        return architecture
    
    @classmethod
    def from_pretrained(cls, chkpt_dir: str, device: str = "cpu") -> tuple["Jepa", dict]:
        """
        Instantiate a JEPA model from a checkpoint and load its weights.
        Assumes the checkpoint structure is the same as created by the Trainer:
        - metadata.json is a dict with a key "architecture" containing the output
            of get_architecture()
        - weights.pt contains the model's state_dict as saved by torch.save.
        """
        metadata_path = os.path.join(chkpt_dir, "metadata.json")
        weights_path = os.path.join(chkpt_dir, "weights.pt")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        architecture = metadata["architecture"]
        encoder = sequential_from_string(architecture["encoder"])
        predictor = sequential_from_string(architecture["predictor"])
        jepa = cls(encoder, predictor)
        jepa.load_state_dict(torch.load(weights_path, map_location=device))
        return jepa, metadata


class JepaCriterion(nn.Module):
    def __init__(
            self,
            re: Optional[nn.Module] = None,
            sparsity_weight: float = 0.0
        ):
        """
        :param re: reconstruction error module (e.g. nn.MSELoss)
        """
        super().__init__()
        if re is None:
            re = nn.MSELoss(reduction="mean")
        self.re = re
        self.sparsity_weight = sparsity_weight

    def forward(self, output: dict, batch: dict) -> dict:
        """
        :param output: dict with keys "encoder_output", "predictor_output", "ema_output"
        :param batch: ignored
        :return: dict with keys "loss", "re", "latent_l1norm"
        """
        ema_output = output["ema_output"]
        encoder_output = output["encoder_output"]
        predictor_output = output["predictor_output"]
        re = self.re(predictor_output, ema_output)
        latent_l1norm = torch.norm(encoder_output, p=1).mean()
        loss = re + self.sparsity_weight * latent_l1norm
        return {"loss": loss, "re": re, "latent_l1norm": latent_l1norm}
    
    def get_config(self) -> dict:
        return {
            "criterion": type(self).__name__,
            "reconstruction_error": type(self.re).__name__,
            "sparsity_weight": self.sparsity_weight
        }


class JepaTrainer(Trainer):
    def __init__(self, alpha: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def train_step(self, batch: dict) -> dict:
        x = batch['x'].to(self.device)
        output = self.model(x)
        losses = self.criterion(output, batch)
        loss = losses["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.update_ema(self.alpha)  # update EMA
        self.step += 1
        if self.log_to_wandb:
            self.log_on_train_step(losses)
        return loss.item()
