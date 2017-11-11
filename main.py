import read
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
import RNN


#input file
text_path = '/Users/thomastiotto/Documents/USI/1 semester/Deep Learning Lab/clarissa_txt/diocane.txt'

# read input text and create dictionary
data, dict, rev_dict = read.read_file(text_path)
dict_size = len(dict)

# Parameters
learning_rate = 0.001
epochs = 2 #epochs
display_step = 1 #expressed in epochs
n_layers = 1
batch_size = 5

# number of units in RNN cell
n_hidden = 512


def encode(x, y):
    x_enc = [dict[i] for i in x]
    y_enc = [dict[y]]

    #x_enc = np.array(x_enc)
    #y_enc = np.array(y_enc)

    return x_enc, y_enc


def train(sess, optimizer, cost, accuracy):
    loss = []
    accuracy = []
    batch_begin = 0 #minibatch start index
    batch_end = batch_size #minibatch end index
    target = 0 #validation target after each minibatch
    n_batches = int(len(data) / batch_size) #number of minibatcheas

    print("dict:", dict)
    print("rev_dict:", rev_dict)

    for j in range(epochs):
        print("----------------------------")
        print("Epoch", j)
        for i in range(n_batches):
            x_batch = data[batch_begin:batch_end]
            y_batch = data[target]

            print("Minibatch",i, ":", x_batch)
            print("Target", y_batch)

            x_batch, y_batch = encode(x_batch, y_batch)

            print("x_enc:\n", x_batch)
            print("y_enc:\n", y_batch)

            _, o, c, a = sess.run([i, optimizer, cost, accuracy], feed_dict={'x:0': x_batch, 'y:0': y_batch})

            if (j % display_step == 0):
                print("Epoch:", j, "Loss:", c, "Accuracy:", a)
                loss.append(c)
                accuracy.append(a)

            batch_begin = i * batch_size
            batch_end = batch_begin + batch_size
            target = batch_end + 1

        i = 0
        batch_begin = 0
        batch_end = 0
        target = 0


def main():

    print("Dataset used:", text_path)
    print("Total number of characters:", len(data))
    print("Number of distinct characters", dict_size)
    print("----------------------------")
    print("Number of epochs:", epochs)
    print("Minibatch size:", batch_size)
    print("Learning rate", learning_rate)
    print("----------------------------")

    sess = tf.InteractiveSession()

    # tf Graph input
    x = tf.placeholder(tf.int64, [None, dict_size], name='x')
    y = tf.placeholder(tf.int64, [None, dict_size], name='y')

    x = tf.one_hot(x, depth=dict_size, on_value=1.0)
    y = tf.one_hot(y, depth=dict_size, on_value=1.0)

    prediction = RNN.RNN(x, y, n_layers, n_hidden, dict_size)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    train(sess, optimizer, cost, accuracy)


if __name__ == "__main__":
    main()