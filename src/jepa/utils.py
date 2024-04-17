import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
import re
import os
import sys


def _extract_params(param_string: str) -> dict:
    if not param_string:
        return [], {}
    positional_params = []
    key_value_params = {}
    pattern = r", (?![^\(\)]*\))"  # split at commas not inside parentheses
    split_result = re.split(pattern, param_string)
    for pair in split_result:
        if len(pair.split("=")) == 1:
            positional = True
            val = pair
        else:
            positional = False
            key, val = pair.split("=")
        try:
            val = eval(val)
        except NameError:
            pass
        if positional:
            positional_params.append(val)
        else:
            key_value_params[key] = val
    # print(positional_params, key_value_params)
    return positional_params, key_value_params


def sequential_from_string(s):
    """
    :param s: the output of str(model), where model is a nn.Sequential object
    :return: the nn.Sequential object
    """
    layers = []
    pattern = r"\((\d+)\): (\w+)\((.*)\)"
    for line in s.split("\n"):
        match = re.match(pattern, line.strip())
        if not match:
            continue
        layer_idx, layer_type, param_string = match.groups()
        # print(layer_idx, layer_type, param_string)
        positional_params, key_value_params = _extract_params(param_string)
        layer_class = getattr(nn, layer_type)
        layer = layer_class(*positional_params, **key_value_params)
        layers.append(layer)
    return nn.Sequential(*layers)


def plot_torch_images(images, title=""):
    """
    :param images: three options:
    - tensor of shape (batch_size, channels, height, width)
    - tensor of shape (channels, height, width)
    - list of tensors of size (channels, height, width)
    """
    gray = False
    if isinstance(images, torch.Tensor):
        match images.ndim:
            case 3:
                images = [images]
            case 2:
                images = [images.unsqueeze(0)]
                gray = True
    nrows = int(len(images) ** 0.5)
    ncols = len(images) // nrows + (len(images) % nrows > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    ravel = np.ravel(axes)
    for i in range(len(images)):
        ax = ravel[i]
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        ax.imshow(image, cmap="gray" if gray else None)
        ax.axis("off")
    fig.suptitle(title)


def have_same_weights(model1: nn.Module, model2: nn.Module) -> bool:
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        try:  # worried about shape mismatch or different layer types
            if p1.data.ne(p2.data).sum() > 0:
                return False
        except Exception as e:  # TODO: catch a more specific exception
            print(e)
            return False
    return True


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
