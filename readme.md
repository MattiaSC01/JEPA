## Installation

To install required packages with pip, navigate to the root directory of the project and run:

```
pip install -r requirements.txt
pip install -e .
```

## Usage

The `scripts` directory contains examples to train an autoencoder as well as a jepa, specifying architecture, optimizer, training hyperparams, logging behaviour, etc. The `notebooks` directory contains a notebook demonstrating how to load pretrained models to perform inference and compute metrics.

## Structure

There are three important objects. A Trainer class (in `trainer/trainer.py`) implements the training and evaluation loop, logging of metrics, and checkpointing. Each type of model (autoencoder, jepa) subclasses Trainer to add model-specific behaviour (`trainer/ae_trainer.py` and `trainer/jepa_trainer.py`). A Logger class (in `logger.py`) provides a simple interface to interact with Weight and Biases during training, logging metrics, images, tables, plots, etc. Finally, each model has a dedicated class subclassing torch.nn.Module with some helpful methods (in `autoencoder.py` and `jepa.py`).

There are utilities to load and preprocess common datasets from torchvision in `dataset.py`, and to perform evaluation of models during and after training in `evaluation.py`, including flatness/denoising capabilities of autoencoders and performance of a linear classifier trained on frozen latents for both autoencoders and jepa.
