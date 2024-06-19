import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

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


# Define a simple neural network class for a single hidden layer
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # Convert predictions to probabilities that sum to one for log_loss
        y_pred = np.hstack([1 - y_pred, y_pred])
        return log_loss(y_true, y_pred)

    def set_weights(self, weights):
        self.weights1 = weights[: self.input_size * self.hidden_size].reshape(
            self.input_size, self.hidden_size
        )
        self.weights2 = weights[self.input_size * self.hidden_size :].reshape(
            self.hidden_size, self.output_size
        )

    def get_weights(self):
        return np.concatenate([self.weights1.flatten(), self.weights2.flatten()])


# Initialize neural network
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1  # Binary classification
nn = SimpleNN(input_size, hidden_size, output_size)


# Define Hill Climbing optimization
def random_neighbor(weights, step_size=0.1):
    neighbor = weights.copy()
    index = np.random.randint(len(weights))
    neighbor[index] += np.random.uniform(-step_size, step_size)
    return neighbor


def hill_climbing(nn, X_train, y_train, max_iter=10000):
    current_weights = nn.get_weights()
    y_pred = nn.forward(X_train)
    current_loss = nn.compute_loss(y_train, y_pred)

    for _ in range(max_iter):
        neighbor_weights = random_neighbor(current_weights)
        nn.set_weights(neighbor_weights)
        y_pred = nn.forward(X_train)
        neighbor_loss = nn.compute_loss(y_train, y_pred)

        if neighbor_loss < current_loss:
            current_weights = neighbor_weights
            current_loss = neighbor_loss

    nn.set_weights(current_weights)
    return nn


# Train the neural network using Hill Climbing
nn_hc = hill_climbing(nn, X_train, y_train, max_iter=10000)

# Evaluate the neural network
y_pred_train = nn_hc.forward(X_train)
y_pred_test = nn_hc.forward(X_test)

# Convert predictions to probabilities for log_loss
y_pred_train_prob = np.hstack([1 - y_pred_train, y_pred_train])
y_pred_test_prob = np.hstack([1 - y_pred_test, y_pred_test])

train_loss = log_loss(y_train, y_pred_train_prob)
test_loss = log_loss(y_test, y_pred_test_prob)

# For binary classification, threshold the predictions at 0.5
train_accuracy = accuracy_score(y_train, (y_pred_train > 0.5).astype(int))
test_accuracy = accuracy_score(y_test, (y_pred_test > 0.5).astype(int))

print(f"Hill Climbing - Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
print(f"Hill Climbing - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
