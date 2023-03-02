import numpy as np

# activation function and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return (np.maximum(0,x))

def relu_prime(x):
    rp = 0
    if x > 0:
        rp = 1
    return rp