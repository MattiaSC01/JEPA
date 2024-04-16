import torch
from torch import nn
from copy import deepcopy
import os
import json
from .utils import sequential_from_string, set_seed


# TODO: create methods to monitor relevant metrics 
#       (e.g. overlap encoder-ema, norms, etc.)


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
        self.update_ema(0)  # copy weights from encoder to ema

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


def get_jepa_loss(sparsity_weight: float = 0.0):
    """
    Build the loss function for the JEPA.
    Needed to have the correct criterion signature for the Trainer.
    """
    def jepa_loss(output: dict, batch: dict):
        """
        :param output: dict with keys "encoder_output", "predictor_output", "ema_output"
        """
        ema_output = output["ema_output"]
        encoder_output = output["encoder_output"]
        predictor_output = output["predictor_output"]
        mse = nn.functional.mse_loss(predictor_output, ema_output)
        if sparsity_weight == 0.0:
            return mse
        sparsity_penalty = sparsity_weight * nn.functional.norm(encoder_output, p=1)
        return mse + sparsity_penalty
    return jepa_loss
