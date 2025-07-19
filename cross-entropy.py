# 3
import numpy as np
# forward pass - input layer - > output layer (includes hidden layers in between)

# 1 x 3 - 3 input neurons
X = np.array([0.5, 0.2, 0.1])

# 3 x 4 weight matrix
# 4 neurons in 2nd layer, each neuron gets 3 weights from input neurons
W1 = np.random.randn(3, 4)
# 1 x 4 biases matrix, each neuron in 2nd layer has a bias        
b1 = np.zeros((1, 4))  

# sigmoid activation function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Z1 - pre activation output matrix - 1 x 4
Z1 = X @ W1 + b1     
# A1 - activation matrix - 1 x 4             
A1 = sigmoid(Z1)                  

# output layer
# 4 x 2 weight matrix - hidden layer neurons to output layer neurons
W2 = np.random.randn(4,2)
# 1 x 2 biases matrix, each neuron in output layer has a bias
b2 = np.zeros((1, 2))

# use softmax for the output layer to get probabilites between 0 and 1
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # prevent overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Z2 - pre activation output matrix - 1 x 2
Z2 = A1 @ W2 + b2
# A2 - activation matrix - 1 x 2      
A2 = softmax(Z2)

def cross_entropy(y_true, y_pred):
    """
    Compute cross-entropy loss - a loss function to make model more accurate

    Parameters:
    - y_true: numpy array of shape (batch_size, num_classes)
              one-hot encoded true labels
    - y_pred: numpy array of shape (batch_size, num_classes)
              predicted probabilities from softmax

    Returns:
    - Scalar loss value (float)
    """
    m = y_true.shape[0]  # number of samples in batch

    # Add epsilon to avoid log(0)
    epsilon = 1e-8
    y_pred_clipped = y_pred + epsilon

    # Compute element-wise loss: -y_true * log(y_pred)
    log_probs = -np.log(y_pred_clipped)
    loss_matrix = y_true * log_probs

    # Sum over all classes, then average across batch
    loss = np.sum(loss_matrix) / m
    return loss

# in this case, since the output matrix is 1 x 2, we have 
# 2 classes, and we're gonna test on class 1, so the 2nd entry in the matrix

y_true = np.array([[0, 1]])  # one-hot: correct label is class 1
loss = cross_entropy(y_true, A2)

# print("Z1 =", Z1)
# print("A1 =", A1)
# print("Z2 =", Z2)
print("A2 (Output Probabilities) =", A2)
print("Cross-Entropy Loss =", loss)