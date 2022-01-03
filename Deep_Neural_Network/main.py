import numpy as np
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from lr_utils import load_dataset


def initialize_parameters_deep(layer_dims):
    parameters = {}
    lens = len(layer_dims)  # number of layers in the network

    for i in range(1, lens):
        parameters['w' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) / np.sqrt(layer_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def linear_forward(a, w, b):
    z = np.dot(w, a) + b
    cache = (a, w, b)
    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    if activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)
    else:
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)
    cache = (linear_cache, activation_cache)
    return a, cache


def l_model_forward(x, parameters):
    caches = []
    a = x
    lens = len(parameters) // 2  # number of layers in the neural network

    for i in range(1, lens):
        a_prev = a
        a, cache = linear_activation_forward(a_prev, parameters['w' + str(i)], parameters['b' + str(i)], "relu")
        caches.append(cache)

    al, cache = linear_activation_forward(a, parameters['w' + str(lens)], parameters['b' + str(lens)], "sigmoid")
    caches.append(cache)
    return al, caches


def compute_cost(al, y):
    m = y.shape[1]
    tem = -(np.dot(np.log(al), y.T) + np.dot(np.log(1 - al), (1 - y.T))) / m
    cost = np.sum(tem)
    return cost


def linear_backward(dz, cache):
    a_pre, w, b = cache
    m = a_pre.shape[1]
    dw = np.dot(dz, a_pre.T) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    da_pre = np.dot(w.T, dz)
    return da_pre, dw, db


def linear_activation_backward(da, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dz = relu_backward(da, activation_cache)
        da_pre, dw,  db = linear_backward(dz, linear_cache)
    else:
        dz = sigmoid_backward(da, activation_cache)
        da_pre, dw, db = linear_backward(dz, linear_cache)

    return da_pre, dw, db


def l_model_backward(al, y, caches):
    grads = {}
    lens = len(caches)
    y = y.reshape(al.shape)
    dal = -(np.divide(y, al) - np.divide(1 - y, 1 - al))
    cac = caches[lens - 1]
    grads['da' + str(lens)], grads['dw' + str(lens)], grads['db' + str(lens)] = \
        linear_activation_backward(dal, cac, 'sigmoid')

    for i in reversed(range(lens - 1)):
        current_cache = caches[i]
        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["da" + str(i + 2)], current_cache,
                                                                    activation="relu")
        grads["da" + str(i + 1)] = da_prev_temp
        grads["dw" + str(i + 1)] = dw_temp
        grads["db" + str(i + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate=0.05):
    lens = len(parameters) // 2

    for i in range(1, lens+1):
        parameters['w' + str(i)] -= learning_rate * grads['dw' + str(i)]
        parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]

    return parameters


def predict(x, parameters):
    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    al, caches = l_model_forward(x, parameters)

    for i in range(al.shape[1]):
        if al[0, i] > 0.5:
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0

    return y_prediction


def deep_nn_model(x, x_test, y, y_test, learning=0.005, iteration=5000):
    features = x.shape[0]
    layer_dims = [features, 20, 10, 5, 1]  # each layer's numbers of nodes
    parameters = initialize_parameters_deep(layer_dims)
    total_cost = []

    for i in range(0, iteration):
        al, caches = l_model_forward(x, parameters)
        grads = l_model_backward(al, y, caches)
        parameters = update_parameters(parameters, grads, learning)
        cost = compute_cost(al, y)
        if i % 20 == 0:
            total_cost.append(cost)
        if i % 200 == 0:
            print('after' + str(i) + 'times cost:', cost)

    y_prediction = predict(x_test, parameters)
    plt.plot(total_cost)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Learning rate:' + str(learning))
    plt.show()
    accuracy = 100 - np.mean(np.abs(y_prediction - y_test) * 100)
    print('The accuracy of test is ' + str(accuracy) + '%')


# load data from datasets
train_set_x_orig, train_y, test_set_x_orig, test_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
train_x = train_set_x_orig.reshape(m_train, -1).T
test_x = test_set_x_orig.reshape(m_test, -1).T
train_x = train_x / 255
test_x = test_x / 255
deep_nn_model(train_x, test_x, train_y, test_y, learning=0.0075, iteration=5000)
