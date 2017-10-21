# Fizz Buzz in Tensorflow!
# largely inspired by http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf
from tf_lib import *
import time

start_perf_counter = time.perf_counter()
start_process_time = time.process_time()


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
    Generate the correct fizz-buzz answer for an integer
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
MAX_INPUT_VALUE = (2 ** (SIZE_INPUT_LAYER + 1)) - 1
TRAINING_RATIO = 2.0 / 3.0
SIZE_OUTPUT_LAYER = 4
SIZE_HIDDEN_LAYERS = 300
NUM_HIDDEN_LAYERS = 1
TRAINING_BATCH_SIZE = 128
TRAINING_ITERATIONS = 4500

ACTIVATION_FUNCTION = tf.nn.relu

EXPECTED_OUTPUT = [
    fizz_buzz_expected_answer(i) for i in range(1, 101)]

# We print the session parameters
print("Session parameters:\n")
print("\tSIZE_INPUT_LAYER\t:", SIZE_INPUT_LAYER)
print("\tMAX_INPUT_VALUE\t\t:", MAX_INPUT_VALUE)
print("\tTRAINING_RATIO\t\t:", TRAINING_RATIO)
print("\tSIZE_OUTPUT_LAYER\t:", SIZE_OUTPUT_LAYER)
print("\tSIZE_HIDDEN_LAYERS\t:", SIZE_HIDDEN_LAYERS)
print("\tNUM_HIDDEN_LAYERS\t:", NUM_HIDDEN_LAYERS)
print("\tTRAINING_BATCH_SIZE\t:", TRAINING_BATCH_SIZE)
print("\tTRAINING_ITERATIONS\t:", TRAINING_ITERATIONS)
print("\tACTIVATION_FUNCTION\t:",
      ACTIVATION_FUNCTION.__module__ + "." + ACTIVATION_FUNCTION.__name__)


# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
# trX = np.array([binary_encode(i, SIZE_INPUT_LAYER)
#                 for i in range(1, MAX_INPUT_VALUE)])
# trY = np.array([fizz_buzz_encode(i) for i in range(1, MAX_INPUT_VALUE)])

trX, trY, selected_numbers = create_training_sets_2(
    SIZE_INPUT_LAYER, MAX_INPUT_VALUE, TRAINING_RATIO, fizz_buzz_encode)

print("\n\nSelected numbers for training (%d numbers):\n"
      % len(selected_numbers))
print(selected_numbers)

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, SIZE_INPUT_LAYER])
Y = tf.placeholder("float", [None, SIZE_OUTPUT_LAYER])


# Predict y given x using the model.
predict_y_given_x = create_prediction_model(
    input_layer=X,
    input_size=SIZE_INPUT_LAYER,
    output_size=SIZE_OUTPUT_LAYER,
    hidden_num=NUM_HIDDEN_LAYERS,
    hidden_size=SIZE_HIDDEN_LAYERS,
    activation_func=ACTIVATION_FUNCTION)

# X, NUM_DIGITS, 4, 2, NUM_HIDDEN)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=predict_y_given_x,
    labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(predict_y_given_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    CHUNK = (TRAINING_ITERATIONS) / 20
    print("\n\nAccuracy on the training data (log every %s iterations):\n"
          % CHUNK)

    for epoch in range(TRAINING_ITERATIONS + 1):
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
        if epoch % CHUNK == 0 or epoch == TRAINING_ITERATIONS:
            print("\t{0: >8}: {1: <3.2f}%".format(
                epoch,
                np.mean(
                    np.argmax(trY, axis=1) == sess.run(
                        predict_op,
                        feed_dict={X: trX, Y: trY})) * 100))

    # And now for some fizz buzz
    numbers = np.arange(1, 1001)
    teX = np.transpose(binary_encode(numbers, SIZE_INPUT_LAYER))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    perf_counter = time.perf_counter() - start_perf_counter
    process_time = time.process_time() - start_process_time
    print("\n\nProcess times:\n")
    print("\tCPU time     : %d sec" % perf_counter)
    print("\tElapsed time : %d sec" % process_time)

    print("\n\nResults:\n")
    errors = 0
    total = 0
    for index, (expected, actual) in enumerate(zip(EXPECTED_OUTPUT, output)):
        errors += 1 if expected != actual else 0
        total += 1
        print("{0: >8}   contained in training: {1: <7}   test passes: {2: >5}   expected: {3: <8}    actual: {4: <8}"
              .format(index + 1,
                      "TRUE" if (index + 1) in selected_numbers else "FALSE",
                      "OK" if expected == actual else "FAIL",
                      expected,
                      actual))

    error_rate = (errors * 100.0) / total
print("\n\nError rate: %2.2f" % error_rate, "%\n")
