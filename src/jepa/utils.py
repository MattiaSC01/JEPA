import torch
from torch import nn
import numpy as np
import random
import hashlib
import re
import os
import sys


def sequential_from_string(s):
    """
    :param s: the output of str(model), where model is a nn.Sequential object
    :return: the nn.Sequential object
    """
    layers = []
    pattern = r"\((\d+)\): (\w+)\((.*?)\)"

    for line in s.split("\n"):
        match = re.match(pattern, line.strip())
        if not match:
            continue
        layer_idx, layer_type, params = match.groups()
        print(layer_idx, layer_type, params)
        param_dict = {}
        if params:
            for pair in params.split(", "):
                key, val = pair.split("=")
                if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):  # catches integers
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                param_dict[key] = val
                print(key, val)

        layer_class = getattr(nn, layer_type)
        layer = layer_class(**param_dict)
        layers.append(layer)

    return nn.Sequential(*layers)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_md5(file_path):
    with open(file_path, 'rb') as file:
        md5_hash = hashlib.md5()
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class Sigmoid(nn.Module):
    """
    implement the function x -> sigmoid(k * x) = 1 / (1 + exp(-k * x))
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.sigmoid(self.k * x)


def clean_wandb():
    """
    Remove all subdirectories in the wandb directory whose name is not "latest-run".
    """
    import os, shutil
    wandb_dir = "wandb"
    assert os.path.exists(wandb_dir), f"wandb directory not found: {wandb_dir}"
    assert os.path.isdir(wandb_dir), f"not a directory: {wandb_dir}"
    for file in os.listdir(wandb_dir):
        if not os.path.isdir(os.path.join(wandb_dir, file)) or file == "latest-run":
            continue
        # scary!
        shutil.rmtree(os.path.join(wandb_dir, file))


class SuppressStdout:
    """
    Context manager to suppress standard output.
    Does not work with jupyter notebooks.
    """
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout
        return False # propagate exceptions
