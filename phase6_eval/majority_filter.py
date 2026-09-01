"""
Follow-up to Phase 6's own observation: "the classification map also shows
visible pixel-level speckle inside otherwise uniform fields -- the expected
signature of classifying each pixel independently with no spatial context."
This applies the standard fix for that -- a post-classification majority
filter -- and checks whether it actually helps.

Reproduces Phase 4/6's exact split and model, gets the full-image prediction
map, then replaces each labeled pixel's prediction with the most common
prediction in its 3x3 neighborhood (background pixels excluded from the
vote). Reports pixel agreement before/after, on the full image and on the
held-out test set specifically.
"""

from scipy.io import loadmat
from scipy import ndimage
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# ======================================
# Load data, reproduce the exact Phase 4/6 split + best model
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

X_all = cube.reshape(-1, 200)
y_all_full = ground_truth.reshape(-1)
all_positions = np.arange(X_all.shape[0])

mask = y_all_full != 0
X = X_all[mask]
y = y_all_full[mask] - 1  # 0-indexed for CrossEntropyLoss
positions = all_positions[mask]  # flat-index of each labeled pixel, for locating the test set spatially later

X_train_full, X_test, y_train_full, y_test, pos_train_full, pos_test = train_test_split(
    X, y, positions, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)


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


print("Retraining Phase 4's winning model (dropout + weight decay, 1000 epochs)...")
model = SpectrumMLP(in_features=200, num_classes=16, dropout_p=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

print("Training complete.\n")

# ======================================
# Full-image prediction map (before filtering)
# ======================================

model.eval()
X_all_scaled = scaler.transform(X_all)
with torch.no_grad():
    all_preds = model(torch.tensor(X_all_scaled, dtype=torch.float32)).argmax(dim=1).numpy()

classification_map = (all_preds + 1).reshape(145, 145)
classification_map[ground_truth == 0] = 0

# ======================================
# 3x3 majority filter: each pixel's label -> most common label among its
# non-background neighbors (including itself)
# ======================================

def majority_vote(window):
    vals = window[window != 0]
    if len(vals) == 0:
        return 0
    return mode(vals, keepdims=False).mode


filtered_map = ndimage.generic_filter(
    classification_map.astype(np.float64), majority_vote, size=3, mode="constant", cval=0
).astype(int)
filtered_map[ground_truth == 0] = 0  # background stays background regardless of neighbor votes

# ======================================
# Accuracy before/after: full image, and the held-out test set specifically
# ======================================

full_mask = ground_truth != 0
acc_full_before = (classification_map == ground_truth)[full_mask].mean()
acc_full_after = (filtered_map == ground_truth)[full_mask].mean()

test_rows, test_cols = np.divmod(pos_test, 145)
gt_test_vals = ground_truth[test_rows, test_cols]
acc_test_before = (classification_map[test_rows, test_cols] == gt_test_vals).mean()
acc_test_after = (filtered_map[test_rows, test_cols] == gt_test_vals).mean()

print(f"Full-image pixel agreement:  before {acc_full_before:.4f}  ->  after {acc_full_after:.4f}")
print(f"Held-out test set accuracy:  before {acc_test_before:.4f}  ->  after {acc_test_after:.4f}")

# ======================================
# Figure
# ======================================

cmap = plt.get_cmap("viridis", 17)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

axes[0].imshow(ground_truth, cmap=cmap, vmin=0, vmax=16)
axes[0].set_title("Ground truth")

axes[1].imshow(classification_map, cmap=cmap, vmin=0, vmax=16)
axes[1].set_title(f"Before filter\n(test acc {acc_test_before:.3f})")

axes[2].imshow(filtered_map, cmap=cmap, vmin=0, vmax=16)
axes[2].set_title(f"After 3x3 majority filter\n(test acc {acc_test_after:.3f})")

plt.tight_layout()
plt.savefig("phase6_eval/majority_filter_comparison.png")
print("\nFigure saved to phase6_eval/majority_filter_comparison.png")
