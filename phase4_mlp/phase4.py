from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

# ======================================
# Load data (same protocol as Phase 3)
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

X = cube.reshape(-1, 200)
y = ground_truth.reshape(-1)

mask = y != 0
X = X[mask]
y = y[mask]

# CrossEntropyLoss expects 0-indexed class labels; ground truth is 1-16
y = y - 1

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)
print("Number of classes:", len(np.unique(y)))

# ======================================
# Train / validation / test split
# ======================================
# Held-out test set: touched only once, at the very end.
# Validation split: carved out of the training set, used every epoch
# to track over/underfitting during training.

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print("\nTrain shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# ======================================
# Standardize (fit on train only)
# ======================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ======================================
# Convert to tensors
# ======================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

NUM_CLASSES = len(np.unique(y))


# ======================================
# Model
# ======================================

class SpectrumMLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ======================================
# Training loop
# ======================================

def train_model(dropout_p, weight_decay, epochs=1000, lr=1e-3):
    torch.manual_seed(42)
    model = SpectrumMLP(in_features=200, num_classes=NUM_CLASSES, dropout_p=dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:3d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}")

    return model, train_losses, val_losses


def test_accuracy(model):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()
    return accuracy_score(y_test, preds)


print("\n=== Training baseline MLP (no regularization) ===")
model_plain, train_losses_plain, val_losses_plain = train_model(dropout_p=0.0, weight_decay=0.0)
acc_plain = test_accuracy(model_plain)
print(f"Baseline test accuracy: {acc_plain:.4f}")

print("\n=== Training regularized MLP (dropout + weight decay) ===")
model_reg, train_losses_reg, val_losses_reg = train_model(dropout_p=0.3, weight_decay=1e-4)
acc_reg = test_accuracy(model_reg)
print(f"Regularized test accuracy: {acc_reg:.4f}")

print(f"\nSVM baseline (Phase 3) test accuracy was 0.8195 for comparison.")

# ======================================
# Plot training curves
# ======================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

best_epoch_plain = int(np.argmin(val_losses_plain))

axes[0].plot(train_losses_plain, label="train loss")
axes[0].plot(val_losses_plain, label="val loss")
axes[0].axvline(best_epoch_plain, color="gray", linestyle="--", linewidth=1,
                 label=f"val loss min (epoch {best_epoch_plain})")
axes[0].set_title(f"No regularization (test acc {acc_plain:.3f})")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].legend()

axes[1].plot(train_losses_reg, label="train loss")
axes[1].plot(val_losses_reg, label="val loss")
axes[1].set_title(f"Dropout + weight decay (test acc {acc_reg:.3f})")
axes[1].set_xlabel("epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("phase4_mlp/training_curve.png")
print("\nTraining curve saved to phase4_mlp/training_curve.png")
