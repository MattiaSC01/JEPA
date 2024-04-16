## Installation

To install required packages with pip, navigate to the root directory of the project and execute:

```
pip install -r requirements.txt
pip install -e .
```

## Usage

The `scripts` directory contains examples to train an autoencoder as well as a jepa, specifying architecture, optimizer, training hyperparams, logging behaviour, etc.
A class Trainer handles the training loop and computation of various metrics. It logs to weights and biases by interacting with a Logger class. Source code is inside `src/jepa`.
