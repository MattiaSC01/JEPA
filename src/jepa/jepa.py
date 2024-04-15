import torch
from torch import nn
from copy import deepcopy


# TODO: Log detailes of the model architecture
# TODO: allow making checkpoints
# TODO: create methods to monitor relevant metrics (e.g. overlap encoder-ema, norms, etc.)


class Jepa(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.ema = deepcopy(encoder)
        for param in self.ema.parameters():
            param.requires_grad = False

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
        :param alpha: momentum for ema update
        """
        for ema_param, param in zip(self.ema.parameters(), self.encoder.parameters()):
            ema_param.copy_((1 - alpha) * param + alpha * ema_param)


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
