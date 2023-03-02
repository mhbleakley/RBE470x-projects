import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_prime
from losses import loss_func, loss_func_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(FCLayer(10, 10))
# net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(loss_func, loss_func_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.75)

# test
out = net.predict(x_train)
print(out)