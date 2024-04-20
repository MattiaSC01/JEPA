from typing import Optional
import torch
from torch import nn
from copy import deepcopy
import os
import json
from ..utils import sequential_from_string, set_seed
from ..evaluation import norm_of_parameters


class Jepa(nn.Module):
    def __init__(
            self,
            encoder: nn.Sequential,
            predictor: nn.Sequential,
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
            assert isinstance(module, nn.Sequential), "encoder and predictor are assumed to be nn.Sequential!"
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, batch: dict) -> dict:
        x = batch["x"]
        x_hat = batch["x_hat"]
        encoder_output = self.encoder(x)
        predictor_output = self.predictor(encoder_output)
        ema_output = self.ema(x_hat)
        return {"encoder_output": encoder_output, "predictor_output": predictor_output, "ema_output": ema_output}
    
    @torch.no_grad()
    def update_ema(self, alpha):
        """
        TODO: make this more efficient: avoid copying the weights, rather update
              them in-place considering momentum and lr of self.encoder. Might 
              require a custom optimizer, or passing the gradients around (ugh).
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
            "type": "Jepa",
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
    
    @torch.no_grad()
    def encode(self, x):
        return self.ema(x)
    
    @torch.no_grad()
    def norms_and_similarities(self) -> dict[str, dict]:
        """
        Compute the cosine similarity between the encoder and the EMA encoder.
        Also compute the L2 norms of the weights of both encoders and of the predictor.
        Consider only the weights of the Linear layers, for now.
        :return: dictionary with keys "similarity", "predictor_norms", "encoder_norms", "ema_norms"
        TODO: consider more general architectures.
        """
        # N.B: it would be more efficient to compute norms and similarity in the same loop,
        # but this way we reuse the norm_of_parameters function and vastly simplify the code.
        predictor_norms = norm_of_parameters(self.predictor)
        encoder_norms = norm_of_parameters(self.encoder)
        ema_norms = norm_of_parameters(self.ema)
        dot_weight, dot_bias = 0.0, 0.0
        for ema_param, param in zip(self.ema.modules(), self.encoder.modules()):
            if not isinstance(ema_param, nn.Linear):
                continue
            assert isinstance(param, nn.Linear), "encoder and ema are assumed to have the same architecture!"
            dot_weight += (ema_param.weight * param.weight).sum()
            dot_bias += (ema_param.bias * param.bias).sum()
        similarity = {}
        similarity["weight_similarity"]= dot_weight / (encoder_norms["weight_norm"] * ema_norms["weight_norm"])
        similarity["bias_similarity"] = dot_bias / (encoder_norms["bias_norm"] * ema_norms["bias_norm"])
        results = {
            "similarity": similarity,
            "predictor_norms": predictor_norms,
            "encoder_norms": encoder_norms,
            "ema_norms": ema_norms,
        }
        return results


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
        latent_l1norm = torch.abs(encoder_output).mean()
        loss = re + self.sparsity_weight * latent_l1norm
        return {"loss": loss, "re": re, "latent_l1norm": latent_l1norm}
    
    def get_config(self) -> dict:
        return {
            "criterion": type(self).__name__,
            "reconstruction_error": type(self.re).__name__,
            "sparsity_weight": self.sparsity_weight
        }
