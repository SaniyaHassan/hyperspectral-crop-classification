from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ======================================
# Load data
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

print("Cube shape:", cube.shape)
print("Ground truth shape:", ground_truth.shape)

# ======================================
# Reshape
# ======================================

X = cube.reshape(-1, 200)
y = ground_truth.reshape(-1)

print("\nAfter reshaping")
print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

# ======================================
# Remove background pixels
# ======================================

mask = y != 0

X = X[mask]
y = y[mask]

print("\nAfter removing background")
print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

# ======================================
# Split into training and testing sets
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("Training label shape:", y_train.shape)
print("Testing label shape:", y_test.shape)

# ======================================
# Standardize the features
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nAfter standardization")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# ======================================
# Train the SVM
# ======================================

model = SVC(kernel="rbf", random_state=42)

model.fit(X_train, y_train)

print("\nSVM training completed.")


# ======================================
# Predictions
# ======================================

predictions = model.predict(X_test)

print("Prediction shape:", predictions.shape)

#Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Test Accuracy:", accuracy)

# ======================================
# Confusion Matrix
# ======================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=90,
    cmap="Blues"
)

plt.title("Confusion Matrix - SVM")
plt.tight_layout()
plt.savefig("phase3_svm/confusion_matrix.png")

print("Confusion matrix saved to phase3_svm/confusion_matrix.png")