import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

from src.jepa.utils import set_seed
from src.jepa.dataset import load_2Dtoy, load_cifar, load_mnist

# fixed hyperparams
# load_dataset = load_cifar
# load_dataset = load_mnist
load_dataset = load_2Dtoy
toy_dataset_type = 'spiral'
toy_noise_scale = 0.0
seed = 42
set_seed(seed)

# dataset
root = "../data"
if load_dataset.__name__ == 'load_2Dtoy':
    train_dataset, test_dataset, train_metadata, test_metadata = load_dataset(
        dataset_type=toy_dataset_type, 
        noise_scale=toy_noise_scale,
        seed=seed
    )
else:
    train_dataset, train_metadata = load_dataset(train=True, log_to_wandb=False, root=root, jepa=True, shuffle=seed)
    test_dataset, test_metadata = load_dataset(train=False, log_to_wandb=False, root=root, jepa=True, shuffle=seed)
train_metadata["use_as"] = "train"
test_metadata["use_as"] = "test"

# Extract data in np for sklearn fitting
x_train, x_test, y_train, y_test = [], [], [], []
for data in train_dataset:
    x, y = data['x'], data['y']
    x_train.append(x)
    y_train.append(y)

for data in test_dataset:
    x, y = data['x'], data['y']
    x_test.append(x)
    y_test.append(y)

x_train, y_train, x_test, y_test = np.vstack(x_train), np.hstack(y_train), \
                                    np.vstack(x_test), np.hstack(y_test)

print('Data shapes')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Transform the data via the covariance matrix
fig, sub = plt.subplots(2, 1)
ax = sub.flatten()
ax[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train)
x_train = x_train - np.mean(x_train, axis=0)
cov_matrix = (x_train.T @ x_train) / x_train.shape[0]
x_train = x_train @ cov_matrix
x_test = (x_test - np.mean(x_test, axis=0)) @ cov_matrix
ax[1].scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.tight_layout()
plt.savefig(f'{load_dataset.__name__}_{toy_dataset_type}_covariance_transform')
plt.close()

# Fit models
models = {'linear': SVC(C=2, kernel='linear', tol=1e-4, random_state=seed), \
        'rbf': SVC(C=2, kernel='rbf', tol=1e-4, random_state=seed)}

if load_dataset.__name__ == 'load_2Dtoy':
    fig, sub = plt.subplots(len(models.keys()), 1)
    ax = sub.flatten()

for i, (name, model) in enumerate(models.items()):
    print('Fitting and evaluating', name, 'model...')
    model.fit(x_train, y_train)
    pred_train, pred_test = model.predict(x_train), model.predict(x_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    print('Train acc:', train_acc, 'test acc:', test_acc)

    # Plot decision boundary (for 2d datasets only)
    if load_dataset.__name__ == 'load_2Dtoy':
        disp = DecisionBoundaryDisplay.from_estimator(
            model,
            x_train,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax[i],
            xlabel='X1',
            ylabel='X2',
        )
        ax[i].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax[i].set_xticks(())
        ax[i].set_yticks(())
        ax[i].set_title(f'{name}')
        print('Done')
        print()

if load_dataset.__name__ == 'load_2Dtoy':
    plt.tight_layout()
    plt.savefig(f'{load_dataset.__name__}_{toy_dataset_type}_decision_boundaries')
    plt.close()