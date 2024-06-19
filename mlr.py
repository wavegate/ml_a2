import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import seaborn as sns
import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose

# Load the dataset
data = pd.read_csv("heart.csv")

# Separate features and target variable
X = data.drop("output", axis=1)
y = data["output"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the sizes of the training and test sets
print(f"Total samples: {len(data)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def run_model(algorithm, X_train, y_train, X_test, y_test, max_iters=1000):
    # Adjust parameters for Simulated Annealing
    if algorithm == "simulated_annealing":
        schedule = mlrose.ExpDecay(
            init_temp=100, exp_const=0.01, min_temp=0.001
        )  # Adjusted exp_const
        model = mlrose.NeuralNetwork(
            hidden_nodes=[10],
            activation="sigmoid",
            algorithm=algorithm,
            max_iters=max_iters,
            bias=True,
            is_classifier=True,
            learning_rate=0.01,
            early_stopping=False,
            clip_max=5,
            max_attempts=100,
            random_state=42,
            curve=True,
            schedule=schedule,
        )
    # Adjust parameters for Genetic Algorithm
    elif algorithm == "genetic_alg":
        model = mlrose.NeuralNetwork(
            hidden_nodes=[10],
            activation="sigmoid",
            algorithm=algorithm,
            max_iters=max_iters,
            bias=True,
            is_classifier=True,
            learning_rate=0.01,
            early_stopping=False,
            clip_max=5,
            max_attempts=100,
            random_state=42,
            curve=True,
            pop_size=200,  # Increased population size
            mutation_prob=0.5,  # Increased mutation rate
        )
    else:
        model = mlrose.NeuralNetwork(
            hidden_nodes=[10],
            activation="sigmoid",
            algorithm=algorithm,
            max_iters=max_iters,
            bias=True,
            is_classifier=True,
            learning_rate=0.01,
            early_stopping=False,
            clip_max=5,
            max_attempts=100,
            random_state=42,
            curve=True,  # Enable fitness curve tracking
        )

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred_train = model.predict(X_train)
    y_prob_train = model.predicted_probs

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predicted_probs

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    train_loss = log_loss(y_train, y_prob_train)
    test_loss = log_loss(y_test, y_prob_test)

    time_elapsed = end_time - start_time
    fitness_curve = model.fitness_curve

    # Make fitness values positive
    min_fitness = np.min(fitness_curve)
    if min_fitness < 0:
        fitness_curve += abs(min_fitness)

    return {
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "time_elapsed": time_elapsed,
        "fitness_curve": fitness_curve,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
    }


# Run models and gather results
algorithms = [
    "random_hill_climb",
    "gradient_descent",
    "simulated_annealing",
    "genetic_alg",
]
results = {}

for algo in algorithms:
    print(f"Running model with {algo}")
    results[algo] = run_model(algo, X_train, y_train, X_test, y_test, max_iters=500)

# Plotting the convergence of loss
plt.figure(figsize=(6, 3))
for algo in algorithms:
    plt.plot(results[algo]["fitness_curve"], label=f"{algo}")

plt.title("Fitness Convergence Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Fitness (Scaled to Positive)")
plt.legend(loc="upper right")
plt.show()

# Plotting Train and Test Accuracy
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].bar(algorithms, [results[algo]["train_accuracy"] for algo in algorithms])
ax[0].set_title("Train Accuracy")
ax[0].set_ylabel("Accuracy")

ax[1].bar(algorithms, [results[algo]["test_accuracy"] for algo in algorithms])
ax[1].set_title("Test Accuracy")
ax[1].set_ylabel("Accuracy")

plt.show()

# Plotting Train and Test Loss
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].bar(algorithms, [results[algo]["train_loss"] for algo in algorithms])
ax[0].set_title("Train Loss")
ax[0].set_ylabel("Loss")

ax[1].bar(algorithms, [results[algo]["test_loss"] for algo in algorithms])
ax[1].set_title("Test Loss")
ax[1].set_ylabel("Loss")

plt.show()

# # Confusion Matrix
# fig, ax = plt.subplots(2, 2, figsize=(14, 12))
# for i, algo in enumerate(algorithms):
#     cm = confusion_matrix(y_test, results[algo]["y_pred_test"])
#     sns.heatmap(cm, annot=True, fmt="d", ax=ax[i // 2, i % 2], cmap="Blues", cbar=False)
#     ax[i // 2, i % 2].set_title(f"Confusion Matrix - {algo}")
#     ax[i // 2, i % 2].set_xlabel("Predicted")
#     ax[i // 2, i % 2].set_ylabel("Actual")

# plt.show()

# # ROC Curve and AUC
# plt.figure(figsize=(12, 6))
# for algo in algorithms:
#     fpr, tpr, _ = roc_curve(y_test, results[algo]["y_prob_test"])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f"{algo} (AUC = {roc_auc:.2f})")

# plt.title("ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="lower right")
# plt.show()

# # Precision-Recall Curve
# plt.figure(figsize=(12, 6))
# for algo in algorithms:
#     precision, recall, _ = precision_recall_curve(y_test, results[algo]["y_prob_test"])
#     plt.plot(recall, precision, label=f"{algo}")

# plt.title("Precision-Recall Curve")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="lower left")
# plt.show()

# Print detailed results
for algo in algorithms:
    print(f"{algo} results:")
    print(f"Train Accuracy: {results[algo]['train_accuracy']}")
    print(f"Test Accuracy: {results[algo]['test_accuracy']}")
    print(f"Train Loss: {results[algo]['train_loss']}")
    print(f"Test Loss: {results[algo]['test_loss']}")
    print(f"Time Elapsed: {results[algo]['time_elapsed']} seconds")
