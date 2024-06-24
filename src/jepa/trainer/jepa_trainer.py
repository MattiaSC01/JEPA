import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from .trainer import Trainer
from ..model.jepa import Jepa
from ..evaluation import build_dataset_of_latents, train_classifier
from ..geom_utils import build_knn_graph, delta_hyperbolicity, oricci_curvature



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
        if isinstance(self.model, Jepa):
            self.model.update_ema(self.alpha) # Exponential moving average weight update for the default jepa model
        return loss
    
    def test_epoch(self):
        loss = super().test_epoch()
        if self.epoch % self.classification_interval == 0:
            self.handle_classification()
        return loss

    def log_gromov_hyperbolicity(self, context_repr, target_repr, step):
        context_graph = build_knn_graph(context_repr)
        target_graph = build_knn_graph(target_repr)
        delta_hyp_context = delta_hyperbolicity(context_graph)
        delta_hyp_target = delta_hyperbolicity(target_graph)
        self.logger.log_metric(delta_hyp_context, "geometric_measures/context_delta_hyperbolicity", step)
        self.logger.log_metric(delta_hyp_target, "geometric_measures/target_delta_hyperbolicity", step)
    
    def log_ricci_curvature(self, context_repr, target_repr, step):
        # Log geometric measures of the latent representations: Gromov delta hyperbolicity, Olivier-Ricci Curvature
        context_graph = build_knn_graph(context_repr)
        target_graph = build_knn_graph(target_repr)
        oricci_context = oricci_curvature(context_graph)
        oricci_target = oricci_curvature(target_graph)

        # log the oricci_curvature as a histogram
        _, axs = plt.subplots(1, 2)
        axs[0].hist(oricci_context, bins=20)
        axs[0].set_title('Context')
        axs[1].hist(oricci_target, bins=20)
        axs[1].set_title('Target')
        self.logger.log_plot("geometric_measures/oricci_curvature", step)
        plt.close()
    
    def log_on_train_step(self, losses):
        super().log_on_train_step(losses)
        # if self.step % (self.log_interval * 20) == 0:
        if self.step % (self.log_interval * 1) == 0:
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
            data = data.detach().cpu().numpy()

            # Log geometric measures of the latent representations: Gromov delta hyperbolicity, Olivier-Ricci Curvature
            self.log_gromov_hyperbolicity(context_repr, target_repr, step=self.step) # TODO: these are not logging frequently
            if self.step % 3 == 0: # log curvature fewer times, heavy calculation (OT dist)
                self.log_ricci_curvature(context_repr, target_repr, step=self.step)

            # When using other datasets, we have to reduce the dim here
            if data.shape[1] > 2:
                data_pca = PCA(n_components=2)
                context_pca = PCA(n_components=2)
                target_pca = PCA(n_components=2)
                data = data_pca.fit_transform(data)
                context_repr = context_pca.fit_transform(context_repr)
                target_repr = target_pca.fit_transform(target_repr)

            _, axs = plt.subplots(1, 3)   
            axs[0].scatter(data[:,0], data[:,1], c=labels)
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
            act_matrix = torch.cat([pred_w, pred_b.unsqueeze(-1)], 1).numpy()
            sns.heatmap(act_matrix, annot=True)
            plt.title('predictor weights and bias (last col)')
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

    @torch.no_grad()
    def build_latent_datasets(self) -> tuple[DataLoader, DataLoader]:
        train_latents = build_dataset_of_latents(self.model.ema, self.train_loader, self.device)
        test_latents = build_dataset_of_latents(self.model.ema, self.test_loader, self.device)
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
