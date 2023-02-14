import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np
from dataset.mnist import load_mnist

class TwoLayerNet :
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01) :
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x) :
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        h1 = sigmoid(a1)
        a2 = np.dot(h1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t) :
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t) :
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t) :
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

def main() :
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True, one_hot_label = True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iters_num = 10000
    train_size = X_train.shape[0]
    batch_size = 1000
    learning_rate = 0.1

    network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num) :
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.numerical_gradient(X_batch, y_batch)

        for key in ('W1', 'b1', 'W2', 'b2') :
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(X_batch, y_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0 :
            train_acc = network.accuracy(X_train, y_train)
            test_acc = network.accuracy(X_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

if __name__ == '__main__':
    main()