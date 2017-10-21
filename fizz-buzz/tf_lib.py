# tf-lib.py
import numpy as np
import tensorflow as tf
from random import randint


def init_weights(shape):
    """
    Initialize the weight in a random fashion
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.001))


def binary_encode(i, num_digits):
    """
    Represent each input by an array of its binary digits.
    """
    return np.array([i >> d & 1 for d in range(num_digits)])


def create_prediction_model(input_layer, input_size, output_size, hidden_num,
                            hidden_size, activation_func=tf.nn.relu):
    """
    Create a prediction model from the imput layer, input/output size,
    hidden layer num/size and an optional activation function (default
    is tensorflow.nn.relu)
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


def create_training_sets(size_input_layer, max_input_value,
                         training_fraction, encoding_function):
    """
    Creates a input training set, output training set and
    the selected number for these sets
    """
    selected_numbers = set()
    if training_fraction > 1:
        training_fraction = 1.0 / training_fraction

    while len(selected_numbers) < max_input_value * training_fraction:
        selected_numbers.add(randint(1, max_input_value))

    train_x = np.array([binary_encode(i, size_input_layer)
                        for i in selected_numbers])
    train_y = np.array([encoding_function(i) for i in selected_numbers])

    return train_x, train_y, selected_numbers


def create_training_sets_2(size_input_layer, max_input_value,
                           training_fraction, encoding_function):
    """
    Creates a input training set, output training set and
    the selected number for these sets
    """
    selected_numbers = set()
    if training_fraction > 1:
        training_fraction = 1.0 / training_fraction

    while len(selected_numbers) < max_input_value * training_fraction:
        selected_numbers.add(randint(1, max_input_value))

    # half = int(max_input_value / 2)
    half = int(len(selected_numbers) / 2)
    l = list(selected_numbers)[0:half]

    train_x = np.array([binary_encode(i, size_input_layer) for i in l])
    train_y = np.array([encoding_function(i) for i in selected_numbers])

    return train_x, train_y, selected_numbers
