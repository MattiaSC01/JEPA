import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import copy
from .dataset import SimpleDataset


@torch.no_grad()
def build_dataset_of_latents(
    encoder: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Dataset:
    """
    Build a dataset of latents produced by an encoder on a dataset
    :param encoder: model to use
    :param data_loader: dataloader spitting dicts with keys 'x', 'y'
    :param device: device to use
    :return: Dataset of latents and labels, spitting dicts with keys 'x', 'y'.
    """
    encoder.to(device).eval()
    latents = []
    labels = []
    for batch in data_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        latents.append(encoder(x))
        labels.append(y)
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    return SimpleDataset(latents, labels)


def train_classifier(
    classifier: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: str = "cpu",
) -> list[float]:
    """
    Train a linear classifier on the provided data.
    :param classifier: model to train. Must output logits.
    :param train_loader: dataloader spitting dicts with keys 'x', 'y'.
    :param test_loader: same as train_loader.
    :param optimizer: Optimizer for training
    :param epochs: number of epochs to train for
    :param device: device to train on
    :return: list of accuracies on the test set after each epoch
    """
    criterion = nn.CrossEntropyLoss()
    classifier.to(device)
    accs = []
    for epoch in range(epochs):
        classifier.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch['x'].to(device), batch['y'].to(device)
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        accs.append(compute_accuracy(classifier, test_loader))
    return accs


@torch.no_grad()
def compute_accuracy(
    classifier: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Compute the accuracy of a classifier on a test set.
    :param classifier: model to evaluate
    :param test_loader: dataloader test data. Expects a dict with keys 'x', 'y'.
    :param device: device to evaluate on
    :return: accuracy
    """
    classifier.to(device).eval()
    correct = 0
    total = 0
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        logits = classifier(x)
        _, predicted = torch.max(logits, 1)  # max returns (values, indices)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    return correct / total


@torch.no_grad()
def norm_of_parameters(model: nn.Module) -> dict:
    """
    Compute the l2 norm of weights and biases (separately)
    of the linear layers of a model. Returns the norms, normalized
    by the number of parameters.
    """
    weight_norm, bias_norm = torch.tensor(0.0), torch.tensor(0.0)
    weight_count, bias_count = 0, 0
    for p in model.modules():
        if not isinstance(p, nn.Linear):
            continue
        weight_norm += (p.weight ** 2).sum()
        bias_norm += (p.bias ** 2).sum()
        weight_count += p.weight.numel()
        bias_count += p.bias.numel()
    weight_norm = weight_norm.sqrt()
    bias_norm = bias_norm.sqrt()
    results = {
        "weight_norm": weight_norm.item(),
        "bias_norm": bias_norm.item(),
        "weight_count": weight_count,
        "bias_count": bias_count,
    }
    return results


class EvalAE:
    """
    This class groups together a bunch of useful static methods for
    evalutation of autoencoders.
    """
    @staticmethod
    @torch.no_grad()
    def evaluate(
            model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            noise_strength: float = 0.0,
            verbose: bool = True,
            data_percentage: float = 1.0,
    ) -> float:
        """
        Evaluate an autoencoder on a test set. Takes care of eval
        mode, no_grad, and moving data to the model's device.
        :param model: The model to evaluate
        :param data_loader: The test set to evaluate on
        :param criterion: criterion(output: dict, batch: dict) -> dict. Output must contain a "loss" key.
        :param noise_strength: Add Gaussian noise with this standard deviation
        to the input data. If nonzero, this evaluates denoising capabilities.
        :param verbose: Whether to log
        :param data_percentage: Use only this percentage of the data. Useful for
        large datasets.
        """
        model.eval()
        device = next(model.parameters()).device
        assert 0.0 <= data_percentage <= 1.0, "data_percentage must be in [0, 1]"
        num_batches = int(data_percentage * len(data_loader))
        test_loss = 0
        for idx, batch in enumerate(data_loader):
            x = batch['x'].to(device)
            noisy_data = EvalAE.corrupt_data(x, noise_strength, noise_type="gaussian-multiplicative")
            output = model(noisy_data)
            test_loss += criterion(output, batch)['loss'].item()
            if idx + 1 >= num_batches:
                break
        test_loss /= num_batches
        if verbose:
            print(f"Test loss: {test_loss:.4f} (noise: {noise_strength})")
        return test_loss

    @staticmethod
    @torch.no_grad()
    def inject_multiplicative_noise(
            model: nn.Module,
            noise_strength: float,
            noise_type: str = "gaussian",
        ) -> None:
        """
        Add multiplicative noise to the weights of all linear layer of a model,
        in-place. Ignores biases.
        :param model: The model to perturb
        :param noise_strength: The strength of the noise to add. For Gaussian noise,
        this is the standard deviation of the noise. For dropout, this is the dropout
        probability.
        :param noise_type: The type of noise to add. Can be "gaussian" or "dropout"
        """
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            match noise_type:
                case "gaussian":
                    mask = torch.ones_like(module.weight) + noise_strength * torch.randn_like(module.weight)
                case "dropout":
                    mask = (torch.rand_like(module.weight) > noise_strength).float()
                case _:
                    raise ValueError(f"Unknown noise type: {noise_type}")
            module.weight.data *= mask

    @staticmethod
    @torch.no_grad()
    def flatness_profile(
            model: nn.Module,
            data_loader: DataLoader,
            noise_strengths: list[float],
            n_iters: int = 10,
            noise_type: str = "gaussian",
            criterion: nn.Module = nn.MSELoss(),
            data_percentage: float = 1.0,
    ) -> dict[list[float]]:
        """
        Estimate the flatness of the loss landscape of a model by perturbing its weights
        and evaluating the loss on a validation set.
        :param model: The model to evaluate
        :param data_loader: The validation set to evaluate on
        :param noise_strengths: Try each of these noise strengths
        :param n_iters: For each noise strength, independently perturb
        the weights iter times.
        :param noise_type: The type of noise to add. Passed to add_noise
        :param criterion: The loss function to use
        :param data_percentage: Use only this percentage of the data.
        """
        losses = defaultdict(list)
        original_state = copy.deepcopy(model.state_dict())
        for noise_strength in noise_strengths:
            for _ in range(n_iters):
                EvalAE.inject_multiplicative_noise(model, noise_strength, noise_type)
                loss = EvalAE.evaluate(
                    model=model, data_loader=data_loader, noise_strength=0.0,
                    criterion=criterion, verbose=False, data_percentage=data_percentage
                )
                losses[noise_strength].append(loss)
                model.load_state_dict(original_state)
        return dict(losses)

    @staticmethod
    @torch.no_grad()
    def denoising_profile(
            model: nn.Module,
            data_loader: DataLoader,
            noise_strengths: list[float],
            n_iters: int = 1,
            criterion: nn.Module = nn.MSELoss(),
            data_percentage: float = 1.0,
    ) -> dict[list[float]]:
        """
        Estimate the denoising capabilities of a model by adding noise to the input
        and evaluating the loss on a validation set.
        :param model: The model to evaluate
        :param data_loader: The validation set to evaluate on
        :param noise_strengths: Try each of these noise strengths
        :param n_iters: Repeat the evaluation on the full dataset n_iters times
        for each noise strength. Useful for small datasets.
        :param criterion: The loss function to use
        :param data_percentage: Use only this percentage of the data.
        """
        losses = defaultdict(list)
        for noise_strength in noise_strengths:
            for _ in range(n_iters):
                loss = EvalAE.evaluate(
                    model=model, data_loader=data_loader, noise_strength=noise_strength,
                    criterion=criterion, verbose=False, data_percentage=data_percentage
                )
                losses[noise_strength].append(loss)
        return dict(losses)

    @staticmethod
    def plot_profile(
            losses: dict[list[float]],
            color: str = "black",
            label: str = "",
    ):
        """
        Plot a flatness or denoising profile.
        :param losses: The losses to plot
        """
        means = [np.mean(losses[noise_strength]) for noise_strength in losses]
        stddevs = [np.std(losses[noise_strength]) for noise_strength in losses]
        plt.errorbar(losses.keys(), means, yerr=stddevs, fmt="o", color=color)
        plt.plot(losses.keys(), means, "-", color=color, label=label)
        plt.xlabel("Noise strength")
        plt.ylabel("Loss")

    @staticmethod
    def corrupt_data(data: torch.Tensor, noise_strength: float, noise_type: str = "gaussian-multiplicative"):
        """
        Corrupt a batch of data with noise.
        :param data: The data to corrupt
        :param noise_strength: The strength of the noise to add
        :param noise_type: The type of noise to add
        """
        match noise_type:
            case "gaussian-multiplicative":
                return data * (torch.ones_like(data) + noise_strength * torch.randn_like(data))
            case "gaussian-additive":
                return data + noise_strength * torch.randn_like(data)
            case "dropout":
                return data * (torch.rand_like(data) > noise_strength).float()
            case "salt-and-pepper":
                return torch.where(
                    torch.rand_like(data) < noise_strength,
                    torch.rand_like(data),
                    data,
                )
            case "identity":
                return data
            case _:
                raise ValueError(f"Unknown noise type: {noise_type}")
