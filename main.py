import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# data setup
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

X = X / 255.0
y = y.astype(np.int8)

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_encoded = one_hot(y)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y_encoded[:60000], y_encoded[60000:]

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# initialize paramaters and functions

# 784 x 128 matrix - 784 weights per hidden layer neuron
W1 = np.random.randn(784, 128) * 0.01
# 1 x 128 matrix - 128 biases for every hidden layer neuron
b1 = np.zeros((1, 128))

# 128 x 10 matrix - 128 weights per neuron in output layer
W2 = np.random.randn(128, 10) * 0.01
# 1 x 10 matrix - 10 biases for every output layer neuron
b2 = np.zeros((1, 10))

# activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# for backprop calc
def sigmoid_derivative(a):
    return a * (1 - a)  

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# forward pass function
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# loss function
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-8 
    y_pred_clipped = y_pred + epsilon
    log_probs = -np.log(y_pred_clipped)
    loss_matrix = y_true * log_probs
    loss = np.sum(loss_matrix) / m
    return loss

# backprop
def backprop(X, y_true, Z1, A1, A2, W2):
    m = X.shape[0] 

    # Output layer gradients
    dZ2 = A2 - y_true                     # shape: (m, 10)
    dW2 = A1.T @ dZ2 / m                  # shape: (128, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # shape: (1, 10)

    # Hidden layer gradients
    dA1 = dZ2 @ W2.T                     # shape: (m, 128)
    dZ1 = dA1 * sigmoid_derivative(A1)  # shape: (m, 128)
    dW1 = X.T @ dZ1 / m                 # shape: (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # shape: (1, 128)

    return dW1, db1, dW2, db2

# accurancy function
def accuracy(X, y_true, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

# Hyperparameters
epochs = 10
learning_rate = 0.1
batch_size = 64

losses = []
# Training loop
for epoch in range(epochs):
    epoch_loss = 0 
    num_batches = 0
    for i in range(0, X_train.shape[0], batch_size):
        # Mini-batch
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        Z1, A1, Z2, A2 = forward(X_batch, W1, b1, W2, b2)

        # Compute loss
        loss = cross_entropy(y_batch, A2)
        epoch_loss += loss
        num_batches += 1

        # Backpropagation
        dW1, db1, dW2, db2 = backprop(X_batch, y_batch, Z1, A1, A2, W2)

        # Gradient descent step
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
        

    # Evaluation on test set after each epoch
    acc = accuracy(X_test, y_test, W1, b1, W2, b2)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc * 100:.2f}%")

# plot loss over time - 
plt.plot(range(1, epochs+1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.grid(True)
plt.show()



