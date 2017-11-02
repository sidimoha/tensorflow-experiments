# foo_bar_qix.py
# Started from https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb

import tempfile
import os
import glob
import numpy as np
from tensorflow.contrib import legacy_seq2seq  # , seq2seq
import tensorflow as tf


# Sequence auto-encoder

write_log = True #False

use_lstm = False
seq_length = 5  # 5
batch_size = 64  # 64
vocab_size = 7
embedding_dim = 50
memory_dim = 300


def single_cell():
    return tf.contrib.rnn.GRUCell(memory_dim)


if use_lstm:
    def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(memory_dim)

tf.reset_default_graph()
# sess = tf.InteractiveSession()
with tf.Session() as sess:

    # seq2seq works generally with tensors, where each
    # tensor represents a single timestep.

    # An imput of to an embedding encoder, for example, would be
    # a list of *seq_length* tensors, each of which is of
    # dimension batch_size (specifying the embedding indices
    # to input at a particular timestep)
    enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                              name="inp%i" % t)
               for t in range(seq_length)]

    labels = [tf.placeholder(tf.int32, shape=(None,),
                             name="labels%i" % t)
              for t in range(seq_length)]

    weights = [tf.ones_like(labels_t, dtype=tf.float32)
               for labels_t in labels]

    # Decoder input: prepend some "GO" token and drop the final
    # token of the encoder input
    dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
               + enc_inp[:-1])

    # Initial memory value for recurrence.
    prev_mem = tf.zeros((batch_size, memory_dim))

    cell = single_cell()

    dec_outputs, dec_memory = legacy_seq2seq.embedding_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell,
        vocab_size,
        vocab_size,
        embedding_size=memory_dim)

    loss = legacy_seq2seq.sequence_loss(
        dec_outputs, labels, weights, vocab_size)
    print("scalar summary:", tf.summary.scalar("loss", loss))

    magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
    print(tf.summary.scalar("magnitude at t=1", magnitude))

    summary_op = tf.summary.merge_all()  # merge_all_summaries()
    print("all summaries:", summary_op)

    # We build the optimizer
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)

    if write_log:
        logdir = "/tmp/foo"
        for fl in glob.glob(logdir + "/*"):
            os.remove(fl)
        #tempfile.mkdtemp()
        print("logdir:", logdir)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        # train.SummaryWriter(logdir, sess.graph_def)

    sess.run(tf.global_variables_initializer())

    def train_batch(batch_size):
        X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
             for _ in range(batch_size)]
        Y = X[:]

        # Dimshuffle to seq_len * batch_size
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

        _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
        return loss_t, summary

    for t in range(500):
        loss_t, summary = train_batch(batch_size)
        if write_log:
            summary_writer.add_summary(summary, t)

    if write_log:
        summary_writer.flush()

    # Basic encoder test
    X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
               for _ in range(10)]
    X_batch = np.array(X_batch).T

    feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
    dec_outputs_batch = sess.run(dec_outputs, feed_dict)

    print("\n\nX_batch:\n")
    print(X_batch)
    print("\n\nOut:")
    print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])
