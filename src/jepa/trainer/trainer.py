from collections import defaultdict
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
from ..logger import WandbLogger
from ..constants import PROJECT, ENTITY
from ..sam import SAM
from ..utils import set_seed
import os
import json
import datetime
import socket
import platform


# TODO: improve how we compute and log validation metrics. Be more systematic
#       about it, e.g. have a list of callable metrics to compute and log.
# TODO: incorporate (optionally? in a subclass?) evaluation through training a linear
#       classifier on top of the encoder.


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        train_metadata: dict,
        test_loader: Optional[DataLoader] = None,
        test_metadata: Optional[dict] = None,
        max_epochs: int = 1,
        device: str = "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_to_wandb: bool = True,
        log_interval: int = 10,
        log_images: bool = False,
        checkpoint_interval: Optional[int] = None,
        checkpoint_root_dir: str = "checkpoints",
        target_loss: Optional[float] = None,
        seed: int = 42,
        compile_model: bool = True,
        is_sweep: bool = False,
        wandb_project: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        validation_interval: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        """
        :param model: pytorch model
        :param optimizer: optimizer
        :param criterion: criterion(output: dict, batch: dict) -> dict. Output must contain a "loss" key.
        :param train_loader: DataLoader that yields batches as dictionaries
        :param test_loader: DataLoader that yields batches as dictionaries
        :param dataset_metadata: metadata about the dataset. Must contain keys 'id', 'use_as'. Can contain other keys.
        :param max_epochs: number of epochs to train for
        :param device: "cpu" or "cuda"
        :param scheduler: learning rate scheduler
        :param log_to_wandb: whether to log to wandb
        :param log_interval: log training loss every log_interval steps
        :param log_images: whether to allow logging images to wandb (heavy!)
        :param checkpoint_interval: save a checkpoint every checkpoint_interval epochs. if None, no checkpoints are saved.
        If None, no checkpoints are saved. To save only at the end of training, set it to epochs.
        :param checkpoint_root_dir: directory to save checkpoints
        :param target_loss: if not None, training stops when the train loss is below this value.
        :param seed: random seed set at the beginning of training.
        :param compile_model: if True, call torch.compile(model) at the end of __init__.
        :param is_sweep: if True, this is a sweep run.
        :param wandb_project: name of the wandb project to log to.
        :param gradient_accumulation_steps: number of steps to accumulate gradients before stepping. Not implemented for SAM.
        :param validation_interval: if not None, do a test epoch every validation_interval steps, in addition to the end of each train epoch.
        :param run_id: a unique identifier of the training run. If not provided, it is set to an integer (avoiding collisions)
        """
        if checkpoint_interval is None:
            checkpoint_interval = max_epochs + 1  # no checkpoints
        else:
            os.makedirs(checkpoint_root_dir, exist_ok=True)
        if target_loss is None:
            target_loss = -float("inf")  # no early stopping
        if wandb_project is None:
            wandb_project = PROJECT
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.max_epochs = max_epochs
        self.device = device
        self.scheduler = scheduler
        self.log_to_wandb = log_to_wandb
        self.log_interval = log_interval
        self.log_images = log_images
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_root_dir = checkpoint_root_dir
        self.target_loss = target_loss
        self.seed = seed
        self.compile_model = compile_model
        self.is_sweep = is_sweep
        self.wandb_project = wandb_project
        self.step = 0
        self.clock = 0
        self.epoch = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_interval = validation_interval
        if run_id is None:
            run_id = str(time.time()).split(".")[0]  # seconds since beginning of time
        self.run_id = run_id
        self.logger = WandbLogger(project=wandb_project, entity=ENTITY)
        self.model.to(self.device)
        if self.compile_model:
            torch.compile(self.model)

    def train_step(self, batch: dict) -> float:
        self.move_to_device(batch)
        output = self.model(batch)
        losses = self.criterion(output, batch)
        loss = losses["loss"]
        loss.backward()
        if self.clock % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.logger.log_metrics()
            self.step += 1
        if self.log_to_wandb:
            self.log_on_train_step(losses)
        self.clock += 1
        return loss.item()

    def train_step_sam(self, batch: dict) -> float:
        if self.gradient_accumulation_steps > 1:
            raise NotImplementedError(
                "Gradient accumulation not implemented for SAM yet."
            )
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        # first forward-backward pass; use original weights w.
        output = self.model(batch)
        losses = self.criterion(output, batch)
        loss = losses["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        # move to w + e(w)
        self.optimizer.first_step()
        # second forward-backward pass; use w + e(w)
        self.optimizer.zero_grad()
        output = self.model(batch)
        self.criterion(output, batch)["loss"].backward()
        # move back to w and use base optimizer to update weights.
        self.optimizer.second_step()
        self.logger.log_metrics()
        self.step += 1
        if self.log_to_wandb:
            self.log_on_train_step(losses)  # log loss from first pass
        return loss.item()

    def log_on_train_step(self, losses):
        """
        :param losses: dictionary with loss values
        """
        if self.step % self.log_interval != 0:
            return
        for key, value in losses.items():
            self.logger.add_metric(value.item(), f"train/{key}", self.step)

    def move_to_device(self, batch: dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], dict):
                for k in batch[key]:
                    if isinstance(batch[key][k], torch.Tensor):
                        batch[key][k] = batch[key][k].to(self.device)
            elif isinstance(batch[key], list):
                for i, item in enumerate(batch[key]):
                    if isinstance(item, torch.Tensor):
                        batch[key][i] = item.to(self.device)

    def train_epoch(self) -> float:
        self.model.train()
        loss = 0.0
        for batch in self.train_loader:
            if self.validation_interval and self.step % self.validation_interval == 0:
                val_loss = self.test_epoch()
                print(f"Step {self.step}   val_loss {val_loss:.4f}")
            if isinstance(self.optimizer, SAM):
                loss += self.train_step_sam(batch)
            else:
                loss += self.train_step(batch)
        self.epoch += 1
        self.logger.add_metric(self.epoch, "train/epoch", self.step)
        self.logger.add_metric(
            self.epoch, "val/epoch", self.step
        )  # redundant, but useful in the dashboard.
        if self.epoch % self.checkpoint_interval == 0:
            self.make_checkpoint()
        return loss / len(self.train_loader)

    def train(self):
        if self.log_to_wandb:
            self.setup_wandb()
        set_seed(self.seed)
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            print(
                f"Epoch {epoch + 1}/{self.max_epochs}   train_loss: {train_loss:.4f}",
                end="",
            )
            if self.test_loader:
                val_loss = self.test_epoch()
                print(f"   val_loss: {val_loss:.4f}", end="")
            print()
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
                print(f"lr: {self.scheduler.get_last_lr()[0]}")
            if train_loss < self.target_loss:
                print(f"Train loss hit target {self.target_loss}. Stopping training.")
                break
        self.end_training()

    def end_training(self):
        if (
            self.checkpoint_interval <= self.max_epochs
            and self.epoch % self.checkpoint_interval != 0
        ):
            self.make_checkpoint()
        if self.log_to_wandb:
            self.logger.end_run()

    def test_step(self, batch: dict) -> dict:
        self.move_to_device(batch)
        output = self.model(batch)
        losses = self.criterion(output, batch)
        return losses

    @torch.no_grad()
    def test_epoch(self) -> float:
        self.model.eval()
        avg_losses = defaultdict(float)
        for batch in self.test_loader:
            losses = self.test_step(batch)
            for key, value in losses.items():
                avg_losses[key] += value.item()
        for key in avg_losses.keys():
            avg_losses[key] /= len(self.test_loader)
        if self.log_to_wandb:
            for key, value in avg_losses.items():
                self.logger.add_metric(value, f"val/{key}", self.step)
        return avg_losses["loss"]

    def make_checkpoint(self):
        # TODO: test changes to this method
        chkpt_dir = os.path.join(
            self.checkpoint_root_dir,
            self.run_id,
            "epoch" + str(self.epoch),
        )
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_path = os.path.join(chkpt_dir, "weights.pt")
        metadata_path = os.path.join(chkpt_dir, "metadata.json")
        val_losses = list(self.logger.get_last_metrics_values(prefix="val/"))
        chkpt_metadata = {
            "step": self.step,
            "epoch": self.epoch,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": self.run_id,
            "optimizer": self.get_optimizer_hyperparameters(),
            "architecture": (
                self.model.get_architecture()
                if hasattr(self.model, "get_architecture")
                else None
            ),
            "hyperparameters": self.get_training_hyperparameters(),
            "device": self.get_device_info(),
            "train_set": self.train_metadata,
            "test_set": self.test_metadata,
        }
        chkpt_metadata["val_losses"] = dict(val_losses)
        with open(metadata_path, "w") as f:
            json.dump(chkpt_metadata, f)
        torch.save(self.model.state_dict(), chkpt_path)
        if self.log_to_wandb:
            artifact_name = f"chkpt-{self.train_metadata['id']}-{self.epoch}"
            self.logger.log_checkpoint(chkpt_dir, artifact_name)

    def setup_wandb(self):
        self.logger.init_run(self.model, is_sweep=self.is_sweep)
        self.logger.use_dataset(self.train_metadata)
        if self.test_metadata:
            self.logger.use_dataset(self.test_metadata)
        self.logger.add_to_config(self.get_training_hyperparameters())
        self.logger.add_to_config(self.get_optimizer_hyperparameters(), prefix="optim")
        self.logger.add_to_config(
            self.get_scheduler_hyperparameters(), prefix="scheduler"
        )
        if hasattr(self.model, "get_architecture"):
            self.logger.add_to_config(self.model.get_architecture(), prefix="model")
        self.logger.add_to_config(self.get_device_info(), prefix="device")
        if self.train_metadata:
            self.logger.add_to_config(self.train_metadata, prefix="train_data")
        if self.test_metadata:
            self.logger.add_to_config(self.test_metadata, prefix="test_data")

    def get_training_hyperparameters(self) -> dict:
        """
        Return a dictionary with the hyperparameters used for training.
        Does not include the model architecture, nor dataset metadata.
        """
        hyperparameters = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "rho": (
                self.optimizer.defaults["rho"]
                if isinstance(self.optimizer, SAM)
                else None
            ),
            "batch_size": self.train_loader.batch_size,
            "max_epochs": self.max_epochs,
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "optimizer": type(self.optimizer).__name__,
            "criterion": (
                self.criterion.get_config()
                if hasattr(self.criterion, "get_config")
                else None
            ),
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "train_size": len(self.train_loader.dataset),
            "test_size": len(self.test_loader.dataset) if self.test_loader else None,
            "target_loss": self.target_loss,
            "seed": self.seed,
            "compile_model": self.compile_model,
        }
        return hyperparameters

    def get_optimizer_hyperparameters(self):
        """
        Return a dictionary with the hyperparameters used for the optimizer.
        """
        d = self.optimizer.defaults
        d["type"] = type(self.optimizer).__name__
        return d

    def get_scheduler_hyperparameters(self):
        """
        Return a dictionary with the hyperparameters used for the scheduler.
        """
        settings = {}
        if not self.scheduler:
            return settings
        for key, value in self.scheduler.__dict__.items():
            if not key.startswith("_") and key not in [
                "optimizer",
                "best",
                "num_bad_epochs",
                "last_epoch",
            ]:
                settings[key] = value
        return settings

    def get_device_info(self):
        """
        Return a dictionary with information about the device used for training.
        """
        device_info = {
            "hostname": socket.gethostname(),
            "cpu": platform.processor(),
            "pytorch_version": torch.__version__,
            "device": self.device,
            "cuda_version": torch.version.cuda,
        }
        return device_info
