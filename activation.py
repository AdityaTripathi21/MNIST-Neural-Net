# 1
import numpy as np

# 1 x 3 - 3 input neurons
X = np.array([0.5, 0.2, 0.1])

# 3 x 4 weight matrix
# 4 neurons in 2nd layer, each neuron gets 3 weights from input neurons
W = np.random.randn(3, 4)
# 1 x 4 biases matrix, each neuron in 2nd layer has a bias        
b = np.zeros((1, 4))  

# sigmoid activation function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Z - pre activation output matrix - 1 x 4
Z = X @ W + b     
# A - activation matrix - 1 x 4             
A = sigmoid(Z)                  

print("Z =", Z)
print("A =", A)