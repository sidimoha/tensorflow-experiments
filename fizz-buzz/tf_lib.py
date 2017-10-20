# tf-lib.py
import numpy as np
import tensorflow as tf
from collections import deque


def init_weights(shape):
    """
    Initialize the weight in a random fashion
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def binary_encode(i, num_digits):
    """
    Represent each input by an array of its binary digits.
    """
    return np.array([i >> d & 1 for d in range(num_digits)])


def create_prediction_model(input_layer, input_size, output_size, hidden_num, hidden_size, activation_func=tf.nn.relu):
    """
    a
    """
    weight_input = init_weights([input_size, hidden_size])
    weight_output = init_weights([hidden_size, output_size])

    weights_hidden = []

    for _ in range(0, hidden_num):
        weights_hidden.append(init_weights([hidden_size, hidden_size]))

    model = activation_func(tf.matmul(input_layer, weight_input))
    for weight in weights_hidden:
        model = activation_func(tf.matmul(model, weight))

    return tf.matmul(model, weight_output)
