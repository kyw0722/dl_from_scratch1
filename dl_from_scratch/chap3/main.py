import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle

def step_function(x) :
    # if x > 0 :
    #     return 1
    # else :
    #     return 0
    # y = x > 0
    #
    # return y.astype(np.int32)
    return np.array(x > 0, dtype = np.int32)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def relu(x) :
    return np.maximum(x, 0)

def identify_func(x) :
    return x

def softmax(x) :
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x

def img_show(img) :
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# def main() :
#     X = np.array([1.0, 0.5])
#     W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     b1 = np.array([0.1, 0.2, 0.3])
#
#     A1 = np.dot(X, W1) + b1
#     h1 = sigmoid(A1)
#
#     W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     b2 = np.array([0.1, 0.2])
#
#     A2 = np.dot(h1, W2) + b2
#     h2 = sigmoid(A2)
#
#     W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
#     b3 = np.array([0.1, 0.2])
#
#     A3 = np.dot(h2, W3) + b3
#     y = identify_func(A3)

def get_data() :
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return X_test, y_test

def init_network() :
    with open("sample_weight.pkl", 'rb') as f :
        network = pickle.load(f)

    return network

def predict(network, X) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    h1 = np.dot(X, W1) + b1
    z1 = sigmoid(h1)
    h2 = np.dot(z1, W2) + b2
    z2 = sigmoid(h2)
    h3 = np.dot(z2, W3) + b3
    y = softmax(h3)

    return y

def main() :
    X, y = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(X), batch_size) :
        X_batch = X[i : i + batch_size]
        y_batch = predict(network, X_batch)
        pred = np.argmax(y_batch, axis = 1)
        accuracy_cnt += np.sum(pred == y[i : i + batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(X)))

if __name__ == '__main__':
    main()