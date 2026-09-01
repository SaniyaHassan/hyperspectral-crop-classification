"""
Re-runs the Phase 4 MLP on a spatially disjoint train/test split instead of
the pixel-random one used everywhere else in this project, to check how much
of the reported accuracy was coming from train and test pixels sitting a few
pixels apart inside the same field.

Method: for each class, find its separate contiguous field patches (connected
components of that class's mask) and assign whole patches to train or test --
never individual pixels. A class that only occupies a single patch can't be
split this way without still leaving train and test touching, so it's kept
entirely in train and excluded from the test-set accuracy here (flagged
below). This "disjoint sampling" concern is a known issue for Indian Pines
specifically -- see Ahmad et al., "Importance of Disjoint Sampling in
Conventional and Transformer Models for Hyperspectral Image Classification"
(arXiv:2404.14944, 2024).
"""

from scipy.io import loadmat
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]

TEST_FRACTION = 0.2

# ======================================
# Load data
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"].astype(np.float64)
ground_truth = gt_data["indian_pines_gt"]

print("Cube shape:", cube.shape)

# ======================================
# Build the spatial split: whole field patches to train or test, per class
# ======================================

rng = np.random.default_rng(42)
struct8 = np.ones((3, 3), dtype=int)  # 8-connectivity, so diagonal-touching pixels count as one field

split_map = np.zeros(ground_truth.shape, dtype=np.uint8)  # 0=background, 1=train, 2=test
excluded_classes = []

print("\nPer-class field patches:")
for class_id in range(1, 17):
    class_mask = ground_truth == class_id
    n_pixels = class_mask.sum()
    if n_pixels == 0:
        continue

    labeled, n_components = ndimage.label(class_mask, structure=struct8)
    component_ids = list(range(1, n_components + 1))
    rng.shuffle(component_ids)

    if n_components == 1:
        split_map[class_mask] = 1
        excluded_classes.append(class_id)
        print(f"  {class_id:2d} {CLASS_NAMES[class_id - 1]:<30s} 1 patch  -> all to train, excluded from test")
        continue

    target_test = n_pixels * TEST_FRACTION
    test_pixels_so_far = 0
    test_ids = set()
    for cid in component_ids:
        comp_size = (labeled == cid).sum()
        if test_pixels_so_far < target_test:
            test_ids.add(cid)
            test_pixels_so_far += comp_size

    for cid in component_ids:
        comp_mask = labeled == cid
        split_map[comp_mask] = 2 if cid in test_ids else 1

    actual_test_frac = test_pixels_so_far / n_pixels
    print(f"  {class_id:2d} {CLASS_NAMES[class_id - 1]:<30s} {n_components} patches -> "
          f"test frac {actual_test_frac:.2f}")

print(f"\nClasses excluded from test set (single contiguous patch): "
      f"{[CLASS_NAMES[c - 1] for c in excluded_classes]}")

# ======================================
# Build train/test feature matrices from the split map
# ======================================

X_all = cube.reshape(-1, 200)
y_all = ground_truth.reshape(-1)
split_all = split_map.reshape(-1)

X_train = X_all[split_all == 1]
y_train = y_all[split_all == 1] - 1  # 0-indexed for CrossEntropyLoss
X_test = X_all[split_all == 2]
y_test = y_all[split_all == 2] - 1

print(f"\nTrain pixels: {X_train.shape[0]}, Test pixels: {X_test.shape[0]}")
print(f"Test set covers {len(np.unique(y_test))} of 16 classes.")

# ======================================
# Standardize (fit on train only), carve a validation split out of train
# ======================================

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_train_sub = scaler.fit_transform(X_train_sub)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_sub, dtype=torch.float32)
y_train_t = torch.tensor(y_train_sub, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

NUM_CLASSES = 16


# ======================================
# Same model + training recipe as Phase 4's winning (regularized) MLP
# ======================================

class SpectrumMLP(nn.Module):
    def __init__(self, in_features, num_classes, dropout_p=0.3):
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


model = SpectrumMLP(in_features=200, num_classes=NUM_CLASSES, dropout_p=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 1000
train_losses, val_losses = [], []

print("\nTraining MLP on the spatially disjoint split...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val_t), y_val_t)
    val_losses.append(val_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch + 1:4d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).argmax(dim=1).numpy()

spatial_acc = accuracy_score(y_test, test_preds)
print(f"\nSpatial-holdout test accuracy: {spatial_acc:.4f}")
print(f"(Phase 4's pixel-random split test accuracy on the same architecture: 0.9290, "
      f"but over all 16 classes -- not a direct apples-to-apples number, see notes.md)")

present_labels = sorted(np.unique(y_test))
present_names = [CLASS_NAMES[i] for i in present_labels]
print("\nPer-class report (classes present in this spatial test set only):")
print(classification_report(y_test, test_preds, labels=present_labels, target_names=present_names, zero_division=0))

# ======================================
# Figure 1: which pixels went where, spatially
# ======================================

cmap = mcolors.ListedColormap(["black", "#4c72b0", "#c44e52"])
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(split_map, cmap=cmap, vmin=0, vmax=2)
ax.set_title("Spatial split: train (blue) vs. test (red) field patches")
cbar = plt.colorbar(im, ax=ax, ticks=[0.33, 1, 1.67])
cbar.ax.set_yticklabels(["background", "train", "test"])
plt.tight_layout()
plt.savefig("spatial_holdout/spatial_split_map.png")
plt.close(fig)
print("\nSplit map saved to spatial_holdout/spatial_split_map.png")

# ======================================
# Figure 2: accuracy comparison against the pixel-random split
# ======================================

fig, ax = plt.subplots(figsize=(5, 4.5))
bars = ax.bar(
    ["Pixel-random split\n(Phase 4, all 16 classes)", "Spatial-holdout split\n(this script)"],
    [0.9290, spatial_acc],
    color=["#4c72b0", "#c44e52"],
)
ax.set_ylim(0, 1)
ax.set_ylabel("Test accuracy")
ax.set_title("Same MLP architecture, different train/test split")
for bar, val in zip(bars, [0.9290, spatial_acc]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
plt.tight_layout()
plt.savefig("spatial_holdout/split_comparison.png")
plt.close(fig)
print("Comparison chart saved to spatial_holdout/split_comparison.png")
