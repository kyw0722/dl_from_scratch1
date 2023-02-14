import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def mean_squared_error(y, t) :
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error_ohe(y, t) :
    if y.ndim == 1 :
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

def numerical_diff(f, x) :
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x) :
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size) :
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100) :
    x = init_x

    for i in range(step_num) :
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def main() :
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True, one_hot_label = True)

    train_size = X_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)

    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

if __name__ == '__main__' :
    main()