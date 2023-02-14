import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *

def main() :
    x = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    for i in range(hidden_layer_size) :
        if i != 0 :
            x = activations[i - 1]

        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
        w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

        a = np.dot(x, w)

        # h = sigmoid(a)
        # h = tanh(a)
        h = relu(a)

        activations[i] = h

    for i, a in activations.items() :
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")

        if i != 0 :
            plt.yticks([], [])
        plt.hist(a.flatten(), 30, range = (0, 1))

    plt.show()

if __name__ == '__main__' :
    main()