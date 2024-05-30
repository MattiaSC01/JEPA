import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from .trainer import Trainer
from ..model.autoencoder import AutoEncoder
from ..evaluation import EvalAE, norm_of_parameters, build_dataset_of_latents, train_classifier
import random


class AutoencoderTrainer(Trainer):
    def __init__(
            self,
            flatness_interval: Optional[int] = None,
            train_set_percentage_for_flatness: Union[float, str] = 1.0,
            flatness_iters: int = 5,
            denoising_iters: int = 1,
            classification_interval: Optional[int] = None,
            classification_epochs: int = 3,
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
        :param classification_interval: train a classifier on latents with this frequency (epochs).
        If None, don't. Requires a test_loader and labelled data.
        :param classification_epochs: number of epochs to train the classifier for.
        """
        super().__init__(**kwargs)
        assert isinstance(self.model, AutoEncoder), "AutoencoderTrainer expects an AutoEncoder model."
        if classification_interval is None:
            classification_interval = self.max_epochs + 1  # no classification
        if flatness_interval is None:
            flatness_interval = self.max_epochs + 1  # no flatness
        if train_set_percentage_for_flatness == 'auto':
            assert self.test_loader is not None, "If train_set_percentage_for_flatness is 'auto', test_loader must be provided."
            train_set_percentage_for_flatness = min(len(self.test_loader.dataset) / len(self.train_loader.dataset), 1.0)
        self.flatness_interval = flatness_interval
        self.train_set_percentage_for_flatness = train_set_percentage_for_flatness
        self.flatness_iters = flatness_iters
        self.denoising_iters = denoising_iters
        self.classification_interval = classification_interval
        self.classification_epochs = classification_epochs

    def log_on_train_step(self, losses):
        super().log_on_train_step(losses)
        if self.step % (self.log_interval * 20) == 0:
            encoder_norms = norm_of_parameters(self.model.encoder)
            decoder_norms = norm_of_parameters(self.model.decoder)
            self.logger.log_metric(encoder_norms["weight_norm"] / encoder_norms["weight_count"], "norms/encoder_weight_norm", self.step)
            self.logger.log_metric(encoder_norms["bias_norm"] / encoder_norms["bias_count"], "norms/encoder_bias_norm", self.step)
            self.logger.log_metric(decoder_norms["weight_norm"] / decoder_norms["weight_count"], "norms/decoder_weight_norm", self.step)
            self.logger.log_metric(decoder_norms["bias_norm"] / decoder_norms["bias_count"], "norms/decoder_bias_norm", self.step)

            # Log representations
            demo_batch = next(iter(self.train_loader))
            data = demo_batch['x'].to(self.device)
            labels = demo_batch['y'].detach().cpu().numpy()
            z = self.model.encoder(data).detach().cpu().numpy()
            data = data.detach().cpu().numpy()
            _, axs = plt.subplots(1, 2)

            # When using other datasets, we have to reduce the dim here
            if data.shape[1] > 2:
                data_pca = PCA(n_components=2)
                context_pca = PCA(n_components=2)
                data = data_pca.fit_transform(data)
                z = context_pca.fit_transform(z)
                
            axs[0].scatter(data[:,0], data[:,1], c=labels)
            axs[0].set_title('Input data')
            axs[1].scatter(z[:,0], z[:,1], c=labels)
            axs[1].set_title('Latent representation')
            plt.tight_layout()
            self.logger.log_plot("repr/encoder_representations", self.step)
            plt.close()
            
    def test_epoch(self):
        loss = super().test_epoch()
        with torch.no_grad():
            if self.epoch % self.flatness_interval == 0:
                self.handle_flatness()
                self.handle_denoising()
            if self.log_to_wandb and self.log_images:
                self.log_image_reconstruction_pairs(split="val")
                self.log_image_reconstruction_pairs(split="train")
        if self.epoch % self.classification_interval == 0:
            self.handle_classification()
        return loss
    
    def end_training(self):
        if self.checkpoint_interval <= self.max_epochs and self.epoch % self.checkpoint_interval != 0:
            self.make_checkpoint()
        if self.flatness_interval <= self.max_epochs and self.epoch % self.flatness_interval != 0:
            self.handle_flatness()
            self.handle_denoising()
        if self.classification_interval <= self.max_epochs and self.epoch % self.classification_interval != 0:
            self.handle_classification()
        if self.log_to_wandb:
            self.logger.end_run()

    def handle_classification(self):
        """
        Use the encoder to create a dataset of latents and train a classifier on it.
        """
        train_dl, test_dl = self.build_latent_datasets()
        accs = self.train_classifier(train_dl, test_dl)
        self.logger.log_metric(max(accs), "classification/best_acc", self.step)
        self.logger.log_metric(accs[-1], "classification/final_acc", self.step)
        if self.is_sweep:
            return
        plt.figure()
        plt.plot(accs)
        plt.title("Test accuracy over epochs - linear classifier on latents")
        self.logger.log_plot("classification/accuracy", self.step)
        plt.close()

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
    
    @torch.no_grad()
    def build_latent_datasets(self) -> tuple[DataLoader, DataLoader]:
        train_latents = build_dataset_of_latents(self.model.encoder, self.train_loader, self.device)
        test_latents = build_dataset_of_latents(self.model.encoder, self.test_loader, self.device)
        train_dl = DataLoader(train_latents, batch_size=self.train_loader.batch_size, shuffle=True)
        test_dl = DataLoader(test_latents, batch_size=self.test_loader.batch_size, shuffle=False)
        return train_dl, test_dl
    
    def train_classifier(self, train_dl: DataLoader, test_dl: DataLoader) -> list:
        x_train, y_train = [], []
        for b in train_dl:
            x_train.append(b['x'].detach().cpu().numpy())
            y_train.append(b['y'].detach().cpu().numpy())
        x_train, y_train = np.vstack(x_train), np.hstack(y_train)

        x_test, y_test = [], []
        for b in test_dl:
            x_test.append(b['x'].detach().cpu().numpy())
            y_test.append(b['y'].detach().cpu().numpy())
        x_test, y_test = np.vstack(x_test), np.hstack(y_test) 

        multiclass_opt = "multinomial" if len(np.unique(y_train)) > 2 else "auto" # For multiclass
        lr = LogisticRegression(C=2, max_iter=10000,  multi_class=multiclass_opt)
        lr.fit(x_train, y_train)
        pred = lr.predict(x_test)
        accs = sum(pred==y_test)/len(y_test)

        return [accs]

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
        self.plot_and_log_flatness(losses, split=split, plot_type="flatness")

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
        self.plot_and_log_flatness(losses, split=split, plot_type="denoising")

    @torch.no_grad()
    def log_image_reconstruction_pairs(self, split: str = "val", n_images: int = 1):
        """
        Log pairs of (noisy) images and their reconstructions to wandb.
        It calls reassemble_image, which for now only works with MNIST and CIFAR.
        Can be turned off by passing log_images=False to the Trainer constructor.
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
                batch = ds[idx]
                batch["x"] = batch["x"].to(self.device)
                noisy = EvalAE.corrupt_data(batch["x"], strength, noise_type=noise_type)
                x_hat = self.model({"x": noisy})["x_hat"]
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
    
    def plot_and_log_flatness(self, losses: dict, split: str, plot_type: str):
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

    def get_training_hyperparameters(self) -> dict:
        hyperparams = super().get_training_hyperparameters()
        hyperparams["train_set_percentage_for_flatness"] = self.train_set_percentage_for_flatness
        return hyperparams
