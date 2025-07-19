# 4

import numpy as np

# ----------- Input Sample (1 x 3) -----------
X = np.array([[0.5, 0.2, 0.1]])
y_true = np.array([[0, 1]])  # One-hot label: class 1 is correct

# ----------- Weights & Biases Initialization -----------
W1 = np.random.randn(3, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2)
b2 = np.zeros((1, 2))

# ----------- Activation Functions -----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# for backprop calc
def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ----------- Loss Function -----------
def cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[0]
    return loss

# ----------- Forward Pass -----------
Z1 = X @ W1 + b1              # (1, 4)
A1 = sigmoid(Z1)              # (1, 4)
Z2 = A1 @ W2 + b2             # (1, 2)
A2 = softmax(Z2)              # (1, 2)
loss = cross_entropy(y_true, A2)

print("Loss before update:", loss)

# ----------- Backpropagation -----------

# essentially, backprop is just chain rule and partial derivatives, and you're finding how sensitive 
# the loss function is to each parameter and then changing the parameter by that much,
# and this process is called gradient descent

# Output layer error (A2 - y_true): shape (1, 2)
dZ2 = A2 - y_true
dW2 = A1.T @ dZ2              # (4, 1) x (1, 2) = (4, 2)
db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, 2)

# Hidden layer error
dA1 = dZ2 @ W2.T              # (1, 2) x (2, 4) = (1, 4)
dZ1 = dA1 * sigmoid_derivative(A1)        # (1, 4)
dW1 = X.T @ dZ1               # (3, 1) x (1, 4) = (3, 4)
db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, 4)

# ----------- Gradient Descent Update -----------

# learning rate defines how big of a step to take when adjusting parameters
learning_rate = 0.1

# under the hood, all parameters in each matrix are being adjusted through a loop 
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W1 -= learning_rate * dW1
b1 -= learning_rate * db1

# ----------- Forward Pass After Update -----------

Z1 = X @ W1 + b1
A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
loss = cross_entropy(y_true, A2)

print("Loss after update:", loss)
