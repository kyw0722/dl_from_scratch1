import numpy as np

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.51
    tmp = w1 * x1 + w2 * x2

    if tmp < theta :
        return 0
    else :
        return 1

def np_AND(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.51
    tmp = b + np.sum(w * x)

    if tmp < 0 :
        return 0
    else :
        return 1

def NAND(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.51
    tmp = b + np.sum(w * x)

    if tmp > 0:
        return 0
    else:
        return 1

def OR(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    tmp = b + np.sum(w * x)

    if tmp <= 0 :
        return 0
    else :
        return 1

def XOR(x1, x2) :
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)

    return y

if __name__ == '__main__':
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))