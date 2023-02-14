import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet :
    def __init__(self) :
        self.W = np.random.randn(2, 3)

    def predict(self, x) :
        return np.dot(x, self.W)

    def loss(self, x, t) :
        h = self.predict(x)
        y = softmax(h)
        loss = cross_entropy_error(y, t)

        return loss

def main() :
    net = simpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)

    t = np.array([0, 0, 1])

    f = lambda w : net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)

if __name__ == '__main__':
    main()