# -*- coding: UTF-8 -*-
'''
这是一个rnn的简单实现，通过模拟两个数的加法，一步步实现rnn，将rnn思想简单直接的展现出来。实现来自文章http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
仅是本人学习过程中的一个记录
'''

import copy
import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# the derivative of sigmoid output
def sigmoid_derivative(output):
    return output * (1 - output)


# generate training dataset
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
# learning rate
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weigjts
weights_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
weights_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
weights_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

weights_0_update = np.zeros_like(weights_0)
weights_1_update = np.zeros_like(weights_1)
weights_h_update = np.zeros_like(weights_h)

# training logic
for j in range(10000):
    # generate a simple addition func c=a+b
    a_int = np.random.randint(largest_number / 2)

    a = int2binary[a_int]
    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we will store our best guess(binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    layer_2_detas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer
        # 这里是rnn精髓，之前的隐含层值传播到当前隐含层，即layer_1_values[-1]
        layer_1 = sigmoid(np.dot(X, weights_0) + np.dot(layer_1_values[-1], weights_h))
        # output layer
        layer_2 = sigmoid(np.dot(layer_1, weights_1))

        layer_2_error = y - layer_2
        layer_2_detas.append(layer_2_error * sigmoid_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_deta = np.zeros(hidden_dim)

    # 反向传播，同时从最后一个时间点往前，根据梯度下降，更新权值
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        pre_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_deta = layer_2_detas[-position - 1]
        # error at hidden layer
        layer_1_deta = (future_layer_1_deta.dot(weights_h.T) + layer_2_deta.dot(weights_1.T)) * sigmoid_derivative(
            layer_1)

        # update weights
        weights_1_update += np.atleast_2d(layer_1).T.dot(layer_2_deta)
        weights_h_update += np.atleast_2d(pre_layer_1).T.dot(layer_1_deta)
        weights_0_update += X.T.dot(layer_1_deta)

        future_layer_1_deta = layer_1_deta

    # 反向传播完成，更新权值
    weights_0 += weights_0_update * alpha
    weights_1 += weights_1_update * alpha
    weights_h += weights_h_update * alpha

    # 重置更新矩阵
    weights_h_update *= 0
    weights_1_update *= 0
    weights_0_update *= 0

    # print output progress
    if (j % 1000 == 0):
        print("Error:", str(overallError))
        print("Pred:", str(d))
        print("True:", str(c))

        out = 0
        for index, x in enumerate(reversed(d)):
            out += pow(2, index) * x
        print(a_int, "+", b_int, "=", out)
