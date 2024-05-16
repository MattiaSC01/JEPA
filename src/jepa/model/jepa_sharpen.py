import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from torch.autograd import Function

from ..utils import sequential_from_string, set_seed
from ..evaluation import norm_of_parameters


class NegateGradient(Function):
    @staticmethod
    def forward(ctx, input):
        return input # indentity map in the forward pass 


    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output # negate the gradient in the backward pass

class nd_act(nn.Module):
    # Acts as an identity activation function, but the gradient is negated (via the function above)
    def forward(self, input):
        return NegateGradient.apply(input)


class JepaSharpen(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            seed: int = 42,
    ):
        super().__init__()
        self.ctx_encoder_w1 = nn.Linear(in_dim, hidden_dim)
        self.ctx_encoder_w2 = nn.Linear(hidden_dim, in_dim)
        self.predictor = nn.Linear(in_dim, in_dim)
        self.delta_wr1 = deepcopy(self.ctx_encoder_w1)
        self.delta_wr2 = deepcopy(self.ctx_encoder_w2)
        self.nd_act = nd_act()
        self.initialize_weights(seed)

    def initialize_weights(self, seed: int):
        # TODO: consider more general architectures (for now, we fall back to default init except for Linear layers)
        set_seed(seed)  # unique seed at the beginning --> change in any module will affect all
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def zero_delta(self):
        with torch.no_grad(): # Set the weights of the perturbation matrix to 0, for gradient evaluation
            for p in self.delta_wr1.parameters():
                p.zero_()
            for p in self.delta_wr2.parameters():
                p.zero_()

    def forward(self, batch: dict) -> dict:
        x = batch["x"]
        x_hat = batch["x_hat"]
        context_out = self.ctx_encoder_w2(F.relu(self.ctx_encoder_w1(x)))
        predictor_out = self.predictor(context_out)
        perturbation = F.relu(self.ctx_encoder_w1(x_hat) + self.delta_wr1(x_hat))
        target_out = self.nd_act(self.ctx_encoder_w2(perturbation) + self.delta_wr2(perturbation)) # Formulation with negated gradient identity map
        
        return {
            "encoder_output": context_out, 
            "predictor_output": predictor_out, 
            "ema_output": target_out
        }
    
    @classmethod
    def from_pretrained(cls, chkpt_dir: str, device: str = "cpu") -> tuple["JepaSharpen", dict]:
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
        latent_code = self.ctx_encoder_w2(F.relu(self.ctx_encoder_w1(x))) # TODO: Do we want to use this for the classification? Effectively, when the model is trained, one cannot perturb so it should be like this.
        return latent_code # We always use this for the classifier, this way the forward function is only used for training.

    @torch.no_grad()
    def context_encode(self, x):
        latent_code = F.relu(self.ctx_encoder_w1(x) + self.delta_wr1(x)) 
        return latent_code 

    @torch.no_grad()
    def norms_and_similarities(self) -> dict[str, dict]:
        """
        Compute the cosine similarity between the encoder and the EMA encoder.
        Also compute the L2 norms of the weights of both encoders and of the predictor.
        Consider only the weights of the Linear layers, for now.
        :return: dictionary with keys "similarity", "predictor_norms", "encoder_norms", "ema_norms"
        TODO: consider more general architectures.
        """
        # TODO: This has to be re-implemented with the current model version
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

