from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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

# ======================================
# Load data
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

X_all = cube.reshape(-1, 200)
y_all = ground_truth.reshape(-1)

mask = y_all != 0
X = X_all[mask]
y = y_all[mask] - 1  # 0-indexed for CrossEntropyLoss, same as Phase 4

# ======================================
# Reproduce the exact Phase 4 split + best model
# (same random_state/architecture/hyperparameters -> the winning
# regularized MLP that beat both the SVM baseline and the unregularized MLP)
# ======================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
# Held-out test set: per-class precision/recall
# ======================================

model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test_scaled, dtype=torch.float32)).argmax(dim=1).numpy()

present_classes = sorted(np.unique(y_test))
report_names = [CLASS_NAMES[c] for c in present_classes]

report = classification_report(
    y_test, test_preds, labels=present_classes, target_names=report_names, digits=3
)
print("Per-class precision/recall (held-out test set):\n")
print(report)

with open("phase6_eval/classification_report.txt", "w") as f:
    f.write(report)
print("Saved to phase6_eval/classification_report.txt")

ConfusionMatrixDisplay.from_predictions(
    y_test, test_preds, xticks_rotation=90, cmap="Blues"
)
plt.title("Confusion Matrix - Regularized MLP (test set)")
plt.tight_layout()
plt.savefig("phase6_eval/mlp_confusion_matrix.png")
plt.close()

# ======================================
# Full-image classification map
# ======================================

X_all_scaled = scaler.transform(X_all)
with torch.no_grad():
    all_preds = model(torch.tensor(X_all_scaled, dtype=torch.float32)).argmax(dim=1).numpy()

# shift back to 1-16 class ids, then zero out background pixels to match
# the ground-truth map's convention (0 = unlabeled)
classification_map = (all_preds + 1).reshape(145, 145)
classification_map[ground_truth == 0] = 0

cmap = plt.get_cmap("viridis", 17)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(ground_truth, cmap=cmap, vmin=0, vmax=16)
axes[0].set_title("Ground Truth")

axes[1].imshow(classification_map, cmap=cmap, vmin=0, vmax=16)
axes[1].set_title("MLP Predicted Classification")

plt.tight_layout()
plt.savefig("phase6_eval/classification_map.png")
plt.close()
print("\nSide-by-side classification map saved to phase6_eval/classification_map.png")

# ======================================
# Error analysis inputs: which classes have worst recall, and are
# their errors spatially "speckled" (isolated misclassified pixels) or
# blocky (whole regions wrong)?
# ======================================

from sklearn.metrics import recall_score

recalls = recall_score(y_test, test_preds, labels=present_classes, average=None)
worst_idx = np.argmin(recalls)
print(f"\nWorst recall: {report_names[worst_idx]} ({recalls[worst_idx]:.3f})")

overall_pixel_agreement = (classification_map == ground_truth)[ground_truth != 0].mean()
print(f"Full-image pixel agreement with ground truth (train+test pixels): {overall_pixel_agreement:.4f}")
