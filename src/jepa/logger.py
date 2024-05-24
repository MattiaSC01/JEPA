import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from collections import defaultdict
from typing import Union, Optional
import numpy as np


# TODO: implement a mechanism to clean up local disk space by deleting old runs in the wandb directory (? - dangerous)


class WandbLogger:
    """
    Handles logging of metrics, images, model checkpoints
    and datasets to Weights and Biases.
    self.metrics contains the history of numerical metrics
    self.to_be_logged contains the metrics that have been added but not logged yet.
    """

    def __init__(
        self,
        project: str,
        entity: str,
    ) -> None:
        self.project = project
        self.entity = entity
        self.metrics_names = set()
        self.last_update_of_metric = {}  # {metric_name: last_step}
        self.metrics = defaultdict(defaultdict(list))  # {step: {metric_name: list_of_values}}
        self.to_be_logged = defaultdict(list)  # {metric_name: list_of_values}

    def init_run(self, model, is_sweep: bool = False, watch_model: bool = True):
        if is_sweep:
            assert (
                wandb.run is not None
            ), "for sweeps, you should call wandb.init() before initializing the logger."
            self.run = wandb.run
        else:
            self.run = wandb.init(
                project=self.project, entity=self.entity, save_code=True, mode="online"
            )
        if watch_model:
            self.run.watch(model, log="all")

    def add_metric(self, value, name: str, step: int):
        print(f"Adding metric {name}: {value}. Step: {step}")
        self.metrics[step][name].append(value)
        self.to_be_logged[name].append(value)
        self.metrics_names.add(name)
        self.last_update_of_metric[name] = step

    def log_metrics(self, step: int):
        metrics = {}
        for name, values in self.to_be_logged.items():
            metrics[name] = np.mean(values).item()
        if metrics:
            self.run.log(metrics, step=step)
            print(f"Logged metrics at step {step}: ", metrics)
        self.to_be_logged = defaultdict(list)  # flush

    def get_last_metrics_values(self, prefix: Optional[str] = None):
        for metric_name in self.metrics_names:
            if prefix is None or metric_name.startswith(prefix):
                if metric_name not in self.last_update_of_metric:
                    continue
                last_step = self.last_update_of_metric[metric_name]
                last_value = self.metrics[last_step][metric_name][-1]
                yield metric_name, last_value

    def log_tensor_as_image(
        self, images: Union[list, torch.Tensor], name: str, step: int
    ):
        """
        :param images: tensor of shape (B, C, H, W) or list
        of tensors of common shape (C, H, W)
        """
        pil_image = self.tensor_to_PIL_image(images)
        self.run.log({name: wandb.Image(pil_image)}, step=step)

    def log_text(self, text: str, name: str):
        self.run.log({name: text})

    def log_plot(self, name: str, step: int):
        self.run.log({name: wandb.Image(plt)}, step=step)

    def log_table(self, df: pd.DataFrame, name: str, step: int):
        """
        There is an issue with the visualization: key "0.01" is displayed as "0\.01".
        :param df: pandas DataFrame
        :param name: name of the table
        :param step: current training step
        """
        wandb_table = wandb.Table(dataframe=df)
        self.run.log({name: wandb_table}, step=step)

    def log_checkpoint(self, chkpt_dir: str, artifact_name: str):
        model_artifact = wandb.Artifact(artifact_name, type="model")
        model_artifact.add_dir(chkpt_dir)
        self.run.log_artifact(model_artifact)

    @staticmethod
    def log_dataset(dataset, metadata: dict, project: str, entity: str):
        """
        Log a dataset as an artifact to Weights and Biases.
        Meant to be called once, after the dataset has been saved to disk.
        :param dataset: dataset to log
        :param metadata: dictionary with metadata about the dataset.
        must contain the following keys: id, dataset_dir (optional).
        :param project: name of the project
        :param entity: name of the entity (e.g. mattia-scardecchia)
        """
        notes = f"Upload dataset {metadata['id']} as an artifact."
        with wandb.init(project=project, entity=entity, notes=notes) as run:
            dataset_artifact = wandb.Artifact(
                name=metadata["id"], type="dataset", metadata=metadata
            )
            if "dataset_dir" in metadata:
                dataset_artifact.add_dir(metadata["dataset_dir"])
            run.log_artifact(dataset_artifact)

    def use_dataset(self, metadata: dict):
        """
        Log to wandb that the dataset has been used for training/testing.
        :param metadata: dictionary with metadata about the dataset.
                         must contain the following keys: id, use_as.
        """
        try:
            self.run.use_artifact(f"{metadata['id']}:latest", use_as=metadata["use_as"])
        except wandb.errors.CommError as e:
            # not sure if this exception is too broad
            print(
                f"Tried to log usage of dataset artifact {metadata['id']}, but it was not found."
            )
            if "mnist" in metadata["id"] or "cifar" in metadata["id"]:
                print(
                    "Continuing without logging the dataset usage, as it is a common dataset."
                )
            else:
                print(
                    "To upload the dataset, include a line like this one in your code:"
                )
                print(
                    "WandbLogger.log_dataset(data, metadata, project=project, entity=ENTITY)"
                )
                raise e

    def add_to_config(self, hyperparameters: dict, prefix: str = ""):
        if prefix:
            hyperparameters = {f"{prefix}/{k}": v for k, v in hyperparameters.items()}
        self.run.config.update(hyperparameters)

    def end_run(self):
        self.run.finish()

    @staticmethod
    @torch.no_grad()
    def tensor_to_PIL_image(images: Union[list, torch.Tensor]) -> Image.Image:
        """
        Convert a tensor to a PIL image using `torchvision.utils.make_grid`
        without normalizing the image.
        :param images: Tensor of shape (B, C, H, W) or list of tensors
        of common shape (C, H, W). expected values in range [-1, 1].
        :return: PIL image ready to be logged
        """
        # If it's a single image, it does nothing; otherwise, it stitches the images together
        grid = torchvision.utils.make_grid(images, normalize=False, nrow=4)
        # go to the [0, 1] range
        grid = (grid + 1) / 2
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        image = Image.fromarray(ndarr)
        return image
