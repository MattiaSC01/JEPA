import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.jepa.utils import set_seed
from src.jepa.dataset import load_2Dtoy
from src.jepa.model.jepa_sharpen import JepaSharpen
from src.jepa.adv_optim import AvdUpd
from src.jepa.logger import WandbLogger

############################## Helper methods ##############################
def log_on_train_step(model, logger, step, loss, latent_l1norm, log_interval, train_loader, device):
        if step % log_interval == 0:
            logger.log_metric(loss.item(), f"train/loss", step)
            logger.log_metric(latent_l1norm.item(), f"train/latent_l1norm", step)

def train_classifier(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(C=2, max_iter=10000)
    lr.fit(x_train, y_train)
    pred = lr.predict(x_test)
    acc = sum(pred==y_test)/len(y_test)

    return acc
####################################################################

# fixed hyperparams
toy_dataset_type = 'spiral'
toy_noise_scale = 0.0
in_dim = 2
hidden_dim = in_dim*10
batch_size = 64
lr = 0.0005
weight_decay = 0.5
max_epochs = 100
gpu_idx = 2
seed = 42
# rho = 0.5
rho = 0.05
# rho = 0.05
# rho = 100

base_optimizer = torch.optim.SGD
# base_optimizer = torch.optim.AdamW
device = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_idx}"
checkpoint_interval = max_epochs # epochs
classification_interval = 10

# Logging
log_interval = 1 # batches
log_to_wandb = True
wandb_project = "jepa-prove"
wandb_entity = "gskenderi"
set_seed(seed)

# dataset
root = "../data"
train_dataset, test_dataset, train_metadata, test_metadata = load_2Dtoy(
    dataset_type=toy_dataset_type, 
    noise_scale=toy_noise_scale,
    seed=seed
)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Data shapes')
print(len(train_dataset), len(test_dataset))


scheduler = None
now = datetime.now()
wandb_run_name = f'sharPPP-rho={rho}-{base_optimizer.__name__}-{toy_dataset_type}--{now}'

# Training objects
model = JepaSharpen(in_dim, hidden_dim, seed).to(device)
param_names = [name for name, _ in model.named_parameters()]
optimizer = AvdUpd(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay, rho=rho, param_names=param_names) # TODO: Currently working with named params here, not sure how optimal this is
criterion = nn.MSELoss()
logger = WandbLogger(project=wandb_project, run_name=wandb_run_name, entity=wandb_entity)
logger.init_run(model)
step = 0

# Training and eval loops
for epoch in range(max_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch in train_loader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        model.zero_delta() # Set the weights of the perturbation matrix to 0, for gradient evaluation
        optimizer.zero_grad()

        # first forward-backward pass; use original weights w.
        output = model(batch)
        ema_output = output["ema_output"]
        encoder_output = output["encoder_output"]
        predictor_output = output["predictor_output"]
        loss = criterion(predictor_output, ema_output)
        latent_l1norm = torch.abs(encoder_output).mean()
        loss.backward()

        # Log representations
        if log_to_wandb:
            if step % log_interval == 0:
                data = batch['x'].to(device)
                labels = batch['y'].detach().cpu().numpy()
                _, axs = plt.subplots(1, 3)
                axs[0].scatter(data[:,0].detach().cpu().numpy(), data[:,1].detach().cpu().numpy(), c=labels)
                axs[0].set_title('Input data')
                axs[1].scatter(ema_output[:,0].detach().cpu().numpy(), ema_output[:,1].detach().cpu().numpy(), c=labels)
                axs[1].set_title('Context')

        #  W_r -> \delta W_r^*
        optimizer.first_step(zero_grad=True)
        output = model(batch)
        ema_output = output["ema_output"]
        encoder_output = output["encoder_output"]
        predictor_output = output["predictor_output"]
        pertb_loss = criterion(predictor_output, ema_output)
        running_loss += pertb_loss.item()
        pertb_loss.backward() # re-populate gradients

        # TODO: Plot representations here (after perturbation)
        if log_to_wandb:
            if step % log_interval == 0:
                axs[2].scatter(ema_output[:,0].detach().cpu().numpy(), ema_output[:,1].detach().cpu().numpy(), c=labels)
                axs[2].set_title('Target')
                plt.tight_layout()
                logger.log_plot("repr/encoder_representations", step)
                plt.close()

        # move back to W_r and use base optimizer to update weights.
        optimizer.second_step()

        if log_to_wandb:
            if step % log_interval == 0:
                # Plot predictor weights here
                pred_w = model.predictor.weight.detach().cpu()
                act_matrix = torch.cat([pred_w], 1).numpy()
                sns.heatmap(act_matrix, annot=True)
                plt.title('predictor weights')
                plt.tight_layout()
                logger.log_plot("repr/predictor_weights", step)
                plt.close()

            # log loss (from first pass)
            log_on_train_step(model, logger, step, loss, latent_l1norm, log_interval, train_loader, device)  
        
        step += 1

    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {running_loss/len(train_loader)}")

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            output = model(batch)
            ema_output = output["ema_output"]
            encoder_output = output["encoder_output"]
            predictor_output = output["predictor_output"]
            vloss = criterion(predictor_output, ema_output)
            logger.log_metric(vloss, "val/loss", step)

        if epoch % classification_interval == 0:
            x_train, y_train = [], []
            for batch in train_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                x_code = model.encode(x)
                x_train.append(x_code.detach().cpu().numpy())
                y_train.append(y.detach().cpu().numpy())
            x_train, y_train = np.vstack(x_train), np.hstack(y_train)

            x_test, y_test = [], []
            for batch in test_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                x_code = model.encode(x)
                x_test.append(x_code.detach().cpu().numpy())
                y_test.append(y.detach().cpu().numpy())
            x_test, y_test = np.vstack(x_test), np.hstack(y_test)

            acc = train_classifier(x_train, y_train, x_test, y_test)
            logger.log_metric(acc, "classification/best_acc", step)
            logger.log_metric(acc, "classification/final_acc", step)
            print(f"Accuracy: {acc}")

logger.end_run()