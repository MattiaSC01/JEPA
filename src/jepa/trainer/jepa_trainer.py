import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

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
        assert isinstance(self.model, Jepa) or isinstance(self.model, JepaAdv), "JepaTrainer expects a Jepa model."
        self.alpha = alpha
        self.classification_interval = classification_interval
        self.classification_epochs = classification_epochs
    
    def train_step(self, batch: dict) -> dict:
        loss = super().train_step(batch)
        if isinstance(self.model, Jepa):
            self.model.update_ema(self.alpha) # Exponential moving average weight update for the default jepa model
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

            # Log representations
            demo_batch = next(iter(self.train_loader))
            data = demo_batch['x'].to(self.device)
            labels = demo_batch['y'].detach().cpu().numpy()
            context_repr = self.model.encoder(data).detach().cpu().numpy()
            target_repr = self.model.ema(data).detach().cpu().numpy()
            _, axs = plt.subplots(1, 3)
            axs[0].scatter(data[:,0].detach().cpu().numpy(), data[:,1].detach().cpu().numpy(), c=labels)
            axs[0].set_title('Input data')
            axs[1].scatter(context_repr[:,0], context_repr[:,1], c=labels)
            axs[1].set_title('Context')
            axs[2].scatter(target_repr[:,0], target_repr[:,1], c=labels)
            axs[2].set_title('Target')
            plt.tight_layout()
            self.logger.log_plot("repr/encoder_representations", self.step)
            plt.close()

            # Log predictor weights
            pred_w, pred_b = self.model.predictor[0].weight.detach().cpu(), self.model.predictor[0].bias.detach().cpu()
            act_matrix = torch.cat([pred_w], 1).numpy()
            sns.heatmap(act_matrix, annot=True)
            plt.title('predictor weights')
            plt.tight_layout()
            self.logger.log_plot("repr/predictor_weights", self.step)
            plt.close()        

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
        train_latents = build_dataset_of_latents(self.model.ema, self.train_loader, self.device)
        test_latents = build_dataset_of_latents(self.model.ema, self.test_loader, self.device)
        train_dl = DataLoader(train_latents, batch_size=self.train_loader.batch_size, shuffle=True)
        test_dl = DataLoader(test_latents, batch_size=self.test_loader.batch_size, shuffle=False)
        return train_dl, test_dl
    
    def train_classifier(self, train_dl: DataLoader, test_dl: DataLoader) -> list:
        # latent_dim = next(iter(train_dl))["x"].shape[1]
        # num_classes = len(train_dl.dataset.labels.unique())
        # classifier = nn.Linear(latent_dim, num_classes).to(self.device)
        # optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
        # criterion = nn.CrossEntropyLoss()
        # accs = train_classifier(classifier=classifier, train_loader=train_dl, test_loader=test_dl, 
        #                         optimizer=optimizer, criterion=criterion, device=self.device, epochs=self.classification_epochs)
        
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

        lr = LogisticRegression(C=2, max_iter=10000)
        lr.fit(x_train, y_train)
        pred = lr.predict(x_test)
        accs = sum(pred==y_test)/len(y_test)

        return [accs]
