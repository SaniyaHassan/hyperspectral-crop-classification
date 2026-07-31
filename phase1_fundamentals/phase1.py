from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

print(type(iris))
print(iris.keys())
print("\nFeature matrix shape:", iris.data.shape)
print("Label vector shape:", iris.target.shape)

print("\nFeature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("Training label shape:", y_train.shape)
print("Testing label shape:", y_test.shape)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("\nTrain prediction shape:", train_predictions.shape)
print("Test prediction shape:", test_predictions.shape)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

shuffle_idx = np.random.RandomState(42).permutation(len(X))
X_shuffled, y_shuffled = X[shuffle_idx], y[shuffle_idx]

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X_shuffled,
    y=y_shuffled,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, test_mean, marker="o", label="Validation Accuracy")

plt.title("Learning Curve - Logistic Regression (Iris)")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.savefig("phase1_fundamentals/learning_curve.png")
print("\nLearning curve saved to phase1_fundamentals/learning_curve.png")