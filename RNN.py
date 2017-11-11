import tensorflow as tf
from tensorflow.contrib import rnn

def RNN(x, y, n_layers, n_hidden, dict_size):

    w = tf.Variable(tf.random_normal([n_hidden, dict_size]))
    b = tf.Variable(tf.random_normal([dict_size]))

    if n_layers==1:
        rnn_cell = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    else:
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)], state_is_tuple=True)

    initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    y, states = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state, dtype=tf.float32)

    y = tf.matmul(y[-1], w) + b

    flat_output = tf.reshape(y, [-1, n_layers])

    return y