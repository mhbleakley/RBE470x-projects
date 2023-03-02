import numpy as np

# loss function and its derivative
def loss_func(y_true, y_pred):
    return (1/2)*np.mean(np.power(y_true-y_pred, 2))

def loss_func_prime(y_true, y_pred):
    return (y_pred-y_true)/y_true.size