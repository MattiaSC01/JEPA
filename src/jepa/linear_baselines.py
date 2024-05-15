import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

from src.jepa.utils import set_seed
from src.jepa.dataset import load_2Dtoy


# fixed hyperparams
toy_dataset_type = 'circle'
toy_noise_scale = 0.0
seed = 42
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

# Fit models
models = {'LR': LogisticRegression(C=2, max_iter=10000, random_state=seed), \
          'SVM': SVC(C=2, kernel='linear', tol=1e-4, random_state=seed)}

fig, sub = plt.subplots(2, 1)
ax = sub.flatten()
for i, (name, model) in enumerate(models.items()):
    print('Fitting and evaluating', name, 'model...')
    model.fit(x_train, y_train)
    pred_train, pred_test = model.predict(x_train), model.predict(x_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    print('Train acc:', train_acc, 'test acc:', test_acc)

    # Plot decision boundary
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

plt.tight_layout()
plt.savefig(f'{toy_dataset_type}_decision_boundaries')
plt.close()