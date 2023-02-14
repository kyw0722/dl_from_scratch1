import numpy as np

def identify_func(x) :
    return x

def step_function(x) :
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def relu(x) :
    return np.maximum(x, 0)

def tanh(x) :
    return np.tanh(x)

def softmax(x) :
    if x.ndim == 2 :
        x = x.T
        c = np.max(x, axis = 0)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x, axis = 0)
        y = exp_x / sum_exp_x
        return y.T

    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x

def mean_squared_error(y, t) :
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size :
        t = t.argmax(axis = 1)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size