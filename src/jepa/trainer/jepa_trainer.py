import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional
from .trainer import Trainer
from ..model.jepa import Jepa
from ..evaluation import build_dataset_of_latents, train_classifier


class JepaTrainer(Trainer):
    def __init__(
            self, 
            alpha: float = 0.99, 
            classification_interval: Optional[int] = None,
            classification_epochs: int = 3,
            **kwargs
        ):
        super().__init__(**kwargs)
        assert isinstance(self.model, Jepa), "JepaTrainer expects a Jepa model."
        self.alpha = alpha
        self.classification_interval = classification_interval
        self.classification_epochs = classification_epochs
    
    def train_step(self, batch: dict) -> dict:
        loss = super().train_step(batch)
        self.model.update_ema(self.alpha)
        return loss
    
    def test_epoch(self):
        loss = super().test_epoch()
        if self.epoch % self.classification_interval == 0:
            self.handle_classification()
        return loss
    
    def log_on_train_step(self, losses):
        super().log_on_train_step(losses)
        if self.step % (self.log_interval * 20) == 0:
            norms = self.model.norms_and_similarities()
            self.logger.log_metric(norms["similarity"]["weight_similarity"], "norms/weight_similarity", self.step)
            self.logger.log_metric(norms["similarity"]["bias_similarity"], "norms/bias_similarity", self.step)
            self.logger.log_metric(norms["predictor_norms"]["weight_norm"] / norms["predictor_norms"]["weight_count"], "norms/predictor_weight_norm", self.step)
            self.logger.log_metric(norms["predictor_norms"]["bias_norm"] / norms["predictor_norms"]["bias_count"], "norms/predictor_bias_norm", self.step)
            self.logger.log_metric(norms["encoder_norms"]["weight_norm"] / norms["encoder_norms"]["weight_count"], "norms/encoder_weight_norm", self.step)
            self.logger.log_metric(norms["encoder_norms"]["bias_norm"] / norms["encoder_norms"]["bias_count"], "norms/encoder_bias_norm", self.step)
            self.logger.log_metric(norms["ema_norms"]["weight_norm"] / norms["ema_norms"]["weight_count"], "norms/ema_weight_norm", self.step)
            self.logger.log_metric(norms["ema_norms"]["bias_norm"] / norms["ema_norms"]["bias_count"], "norms/ema_bias_norm", self.step)

    def end_training(self):
        if self.checkpoint_interval <= self.max_epochs and self.epoch % self.checkpoint_interval != 0:
            self.make_checkpoint()
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

    @torch.no_grad()
    def build_latent_datasets(self) -> tuple[DataLoader, DataLoader]:
        train_latents = build_dataset_of_latents(self.model.encoder, self.train_loader, self.device)
        test_latents = build_dataset_of_latents(self.model.encoder, self.test_loader, self.device)
        train_dl = DataLoader(train_latents, batch_size=self.train_loader.batch_size, shuffle=True)
        test_dl = DataLoader(test_latents, batch_size=self.test_loader.batch_size, shuffle=False)
        return train_dl, test_dl
    
    def train_classifier(self, train_dl: DataLoader, test_dl: DataLoader) -> list:
        latent_dim = next(iter(train_dl))["x"].shape[1]
        num_classes = len(train_dl.dataset.labels.unique())
        classifier = nn.Linear(latent_dim, num_classes)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        accs = train_classifier(classifier=classifier, train_loader=train_dl, test_loader=test_dl, 
                                optimizer=optimizer, criterion=criterion, device=self.device, epochs=self.classification_epochs)
        return accs
