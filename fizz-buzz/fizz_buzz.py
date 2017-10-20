# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf
from tf_lib import *


def fizz_buzz_encode(i):
    """
    encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
    """
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


def fizz_buzz(i, prediction):
    """
    turn a prediction (and an original number)
    into a fizz buzz output
    """
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def fizz_buzz_expected_answer(i):
    """
    Generate the correct fizz-buzz answer for an int
    """
    if i % 15 == 0:
        return "fizzbuzz"
    elif i % 5 == 0:
        return "buzz"
    elif i % 3 == 0:
        return "fizz"
    else:
        return str(i)


SIZE_INPUT_LAYER = 10
SIZE_OUTPUT_LAYER = 4
SIZE_HIDDEN_LAYERS = 300
NUM_HIDDEN_LAYERS = 1
TRAINING_BATCH_SIZE = 128

# ITER | SIZE_HL | NUM_HL
# 6500 | 1       | 100
# 3000 | 2       | 100
# 1800 | 1       | 200
# 1500 | 1       | 300
TRAINING_ITERATIONS = 1500

ACTIVATION_FUNCTION = tf.nn.relu

EXPECTED_OUTPUT = [
    fizz_buzz_expected_answer(i) for i in range(1, 101)]


# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, SIZE_INPUT_LAYER)
                for i in range(1, 2 ** SIZE_INPUT_LAYER)])
trY = np.array([fizz_buzz_encode(i) for i in range(1, 2 ** SIZE_INPUT_LAYER)])

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, SIZE_INPUT_LAYER])
Y = tf.placeholder("float", [None, SIZE_OUTPUT_LAYER])


# Predict y given x using the model.
# predict_y_given_x = model(X, w_h, w_i, w_o)
predict_y_given_x = create_prediction_model(
    input_layer=X,
    input_size=SIZE_INPUT_LAYER,
    output_size=SIZE_OUTPUT_LAYER,
    hidden_num=NUM_HIDDEN_LAYERS,
    hidden_size=SIZE_HIDDEN_LAYERS,
    activation_func=ACTIVATION_FUNCTION)

# X, NUM_DIGITS, 4, 2, NUM_HIDDEN)

# We'll train our model by minimizing a cost function.
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=predict_y_given_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(predict_y_given_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())

    for epoch in range(TRAINING_ITERATIONS):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of TRAINING_BATCH_SIZE inputs.
        for start in range(0, len(trX), TRAINING_BATCH_SIZE):
            end = start + TRAINING_BATCH_SIZE
            sess.run(
                train_op,
                feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # And now for some fizz buzz
    numbers = np.arange(1, 1001)
    teX = np.transpose(binary_encode(numbers, SIZE_INPUT_LAYER))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print("result:")
    for index, (expected, actual) in enumerate(zip(EXPECTED_OUTPUT, output)):
        # print(index, "expected:", expected, "actual:", actual)
        print("integer: {0: >4} {1: <6} {2: <9} {3: <8} {4: <9} {5: <8}".format(
            index + 1, "  OK  " if expected == actual else " FAIL ", "expected:", expected, "actual:", actual))
