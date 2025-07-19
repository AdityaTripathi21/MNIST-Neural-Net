# 2
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


print("Z1 =", Z1)
print("A1 =", A1)
print("Z2 =", Z2)
print("A2 (Output Probabilities) =", A2)