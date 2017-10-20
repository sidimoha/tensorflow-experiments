# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf

NUM_DIGITS = 10


def binary_encode(i, num_digits):
    """
    Represent each input by an array of its binary digits.
    """
    return np.array([i >> d & 1 for d in range(num_digits)])


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


def init_weights(shape):
    """
    Initialize the weight in a random fashion
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def fizz_buzz(i, prediction):
    """
    turn a prediction (and an original number)
    into a fizz buzz output
    """
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def model(input_layer, w_h, w_i, w_o):
    """
    Create the Neural Network with 2 hidden layer (perceptron style) with ReLU activation function
    """
    h = tf.nn.relu(tf.matmul(input_layer, w_h))
    h_i = tf.nn.relu(tf.matmul(h, w_i))
    return tf.matmul(h_i, w_o)


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


EXPECTED_OUTPUT = [
    fizz_buzz_expected_answer(i) for i in range(1, 101)]


# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, NUM_DIGITS)
                for i in range(1, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i) for i in range(1, 2 ** NUM_DIGITS)])

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_i = init_weights([NUM_HIDDEN, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

# Predict y given x using the model.
py_x = model(X, w_h, w_i, w_o)

# We'll train our model by minimizing a cost function.
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

BATCH_SIZE = 128
# 6500 with 1 hidden layer
# 3000 with 2 hidden layers
TRAINING_ITERATIONS = 3000

# Launch the graph in a session
with tf.Session() as sess:
    # tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())

    for epoch in range(TRAINING_ITERATIONS):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={
                     X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        # print(epoch, np.mean(np.argmax(trY, axis=1) ==
        #                      sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # And now for some fizz buzz
    numbers = np.arange(1, 1001)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print("result:")
    for index, (expected, actual) in enumerate(zip(EXPECTED_OUTPUT, output)):
        # print(index, "expected:", expected, "actual:", actual)
        if expected != actual:
            print("integer: {0: >4} {1: <6} {2: <9} {3: <8} {4: <9} {5: <8}".format(
                index + 1, "  OK  " if expected == actual else " FAIL ", "expected:", expected, "actual:", actual))
    else:
        print("All OK")
