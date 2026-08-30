"""
Quantum machine learning extension: builds a small variational quantum
classifier (PennyLane) on PCA-reduced features from the Indian Pines cube,
and compares it against a classical baseline on the identical features/split.

Scope choice: binary classification between the two largest classes
(Soybean-mintill, Corn-notill) rather than full 16-class. A single
qubit's Z-expectation naturally gives one binary decision boundary;
extending to 16-way classification would need either 4 output qubits
read out jointly or a one-vs-rest ensemble of these circuits, which is
a real next step but out of scope for what's buildable here.
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import time

torch.manual_seed(42)
np.random.seed(42)

CLASS_NAMES = {2: "Corn-notill", 11: "Soybean-mintill"}

# ======================================
# Load data, restrict to the two largest classes
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

X_all = cube.reshape(-1, 200)
y_all = ground_truth.reshape(-1)

mask = np.isin(y_all, list(CLASS_NAMES.keys()))
X = X_all[mask]
y = y_all[mask]

print("Samples per class:")
for class_id, name in CLASS_NAMES.items():
    print(f"  {name}: {(y == class_id).sum()}")

# Subsample for tractable quantum-circuit simulation runtime -- 150 per
# class train, 50 per class test. A quantum simulator on a laptop CPU
# doesn't scale to the full ~3900-sample set the way a classical SVM does;
# this is itself a finding worth reporting, not just a shortcut.
N_TRAIN_PER_CLASS = 150
N_TEST_PER_CLASS = 50

rng = np.random.default_rng(42)
train_idx, test_idx = [], []
for class_id in CLASS_NAMES:
    class_indices = np.where(y == class_id)[0]
    rng.shuffle(class_indices)
    train_idx.extend(class_indices[:N_TRAIN_PER_CLASS])
    test_idx.extend(class_indices[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_TEST_PER_CLASS])

train_idx = np.array(train_idx)
test_idx = np.array(test_idx)
rng.shuffle(train_idx)
rng.shuffle(test_idx)

X_train_raw, y_train_raw = X[train_idx], y[train_idx]
X_test_raw, y_test_raw = X[test_idx], y[test_idx]

# Binary labels: +1 / -1 for the two classes (needed for the quantum
# circuit's Z-expectation readout, which naturally lives in [-1, 1])
label_map = {2: -1, 11: 1}
y_train = np.array([label_map[v] for v in y_train_raw])
y_test = np.array([label_map[v] for v in y_test_raw])

print(f"\nTrain samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

# ======================================
# Feature prep: standardize -> PCA to 4 components (angle-encoded onto
# 4 qubits) -> scale to [-pi, pi] for rotation-gate angles
# ======================================

N_QUBITS = 4

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_raw)
X_test_std = scaler.transform(X_test_raw)

pca = PCA(n_components=N_QUBITS, random_state=42)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(f"PCA({N_QUBITS}) explained variance: {pca.explained_variance_ratio_.sum():.1%}")

angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_train_angles = angle_scaler.fit_transform(X_train_pca)
X_test_angles = angle_scaler.transform(X_test_pca)

# ======================================
# Classical baseline, same exact features/split, for a fair comparison
# ======================================

clf = SVC(kernel="rbf", random_state=42)
t0 = time.time()
clf.fit(X_train_pca, y_train)
classical_train_time = time.time() - t0
classical_acc = accuracy_score(y_test, clf.predict(X_test_pca))
print(f"\nClassical SVM (same 4 PCA features): {classical_acc:.4f} "
      f"(train time {classical_train_time:.3f}s)")

# ======================================
# Variational quantum classifier
# ======================================

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


N_LAYERS = 3
weights = torch.tensor(
    0.01 * np.random.randn(N_LAYERS, N_QUBITS), requires_grad=True, dtype=torch.float64
)
bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

# Carve a validation set out of the training data (distinct from the test
# set) so "best epoch" is picked honestly -- same discipline as Phase 4's
# MLP, not a peek at test-set performance.
X_tr_angles, X_val_angles, y_tr, y_val = train_test_split(
    X_train_angles, y_train, test_size=0.2, random_state=42, stratify=y_train
)

X_tr_t = torch.tensor(X_tr_angles, dtype=torch.float64)
y_tr_t = torch.tensor(y_tr, dtype=torch.float64)
X_val_t = torch.tensor(X_val_angles, dtype=torch.float64)
y_val_t = torch.tensor(y_val, dtype=torch.float64)
X_test_t = torch.tensor(X_test_angles, dtype=torch.float64)
y_test_t = torch.tensor(y_test, dtype=torch.float64)

optimizer = torch.optim.Adam([weights, bias], lr=0.05)


def predict(weights, bias, X):
    return torch.stack([circuit(weights, x) for x in X]) + bias


def accuracy(preds, labels):
    return ((torch.sign(preds) == labels).float().mean()).item()


EPOCHS = 40
train_losses, train_accs, val_accs = [], [], []
best_val_acc = -1.0
best_weights, best_bias, best_epoch = None, None, -1

print(f"\nTraining variational quantum classifier ({N_QUBITS} qubits, {N_LAYERS} layers)...")
t0 = time.time()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    preds = predict(weights, bias, X_tr_t)
    loss = torch.mean((preds - y_tr_t) ** 2)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_acc = accuracy(preds.detach(), y_tr_t)
    train_accs.append(train_acc)

    with torch.no_grad():
        val_preds = predict(weights, bias, X_val_t)
        val_acc = accuracy(val_preds, y_val_t)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = weights.detach().clone()
        best_bias = bias.detach().clone()
        best_epoch = epoch

    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch + 1:2d}  loss {loss.item():.4f}  "
              f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

quantum_train_time = time.time() - t0

# Evaluate the best-validation-accuracy checkpoint on the held-out test
# set exactly once.
with torch.no_grad():
    final_test_preds = predict(best_weights, best_bias, X_test_t)
    quantum_acc = accuracy(final_test_preds, y_test_t)

print(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch + 1}")
print(f"Quantum classifier test accuracy (best-val checkpoint): {quantum_acc:.4f} "
      f"(train time {quantum_train_time:.1f}s for {EPOCHS} epochs)")
print(f"Classical SVM test accuracy:                            {classical_acc:.4f} "
      f"(train time {classical_train_time:.3f}s)")
print(f"\nSpeed ratio: quantum classifier took "
      f"{quantum_train_time / max(classical_train_time, 1e-6):.0f}x longer to train "
      f"on this CPU simulator.")

# ======================================
# Plot training curve + final comparison
# ======================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(train_accs, label="train accuracy")
axes[0].plot(val_accs, label="val accuracy")
axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                 label=f"best val (epoch {best_epoch + 1})")
axes[0].set_xlabel("epoch")
axes[0].set_title("Quantum classifier training")
axes[0].legend()

bars = axes[1].bar(
    ["Classical SVM\n(4 PCA features)", f"Quantum VQC\n({N_QUBITS} qubits)"],
    [classical_acc, quantum_acc],
    color=["#4c72b0", "#c44e52"],
)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Corn-notill vs. Soybean-mintill")
for bar, val in zip(bars, [classical_acc, quantum_acc]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")

plt.tight_layout()
plt.savefig("quantum_extension/quantum_vs_classical.png")
print("\nFigure saved to quantum_extension/quantum_vs_classical.png")
