from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time

# ======================================
# Reproducing the protocol from:
# Ustuner, M. "Randomized Principal Component Analysis for Hyperspectral
# Image Classification." 2024 IEEE M2GARSS.
# -> reduce features to 20 and 30 principal components, then classify
#    with an SVM, on Indian Pines (among other datasets).
# See notes.md for citation details and critique.
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# Baseline: all 200 bands, no PCA (this is Phase 3's SVM, rerun here for
# a same-script, apples-to-apples comparison point)
model_full = SVC(kernel="rbf", random_state=42)
model_full.fit(X_train_scaled, y_train)
acc_full = accuracy_score(y_test, model_full.predict(X_test_scaled))
results["200 bands (no PCA)"] = acc_full
print(f"200 bands (no PCA): {acc_full:.4f}")

# Paper's protocol: reduce to 20 and 30 principal components
explained_variance = {}
for n_components in [20, 30]:
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_variance[n_components] = pca.explained_variance_ratio_.sum()

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train_pca, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_pca))
    results[f"{n_components} PCA components"] = acc
    print(
        f"{n_components} PCA components "
        f"(explains {explained_variance[n_components]:.1%} of variance): {acc:.4f}"
    )

# ======================================
# Plot: accuracy retained vs. dimensionality reduction
# ======================================

labels = list(results.keys())
values = list(results.values())

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452", "#55a868"])
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1)
ax.set_title("SVM accuracy: full spectrum vs. PCA-reduced features")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
             ha="center")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("phase7_paper/pca_svm_comparison.png")
print("\nFigure saved to phase7_paper/pca_svm_comparison.png")
