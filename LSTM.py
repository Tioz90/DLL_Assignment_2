import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28 #1
n_chunks = 28 #1
rnn_size = 128

sess = tf.InteractiveSession()

x = tf.placeholder('float', [None, n_chunks, chunk_size])
# x = tf.placeholder('float', [batch_size, chunk_size, n_classes])
y = tf.placeholder('float')
# x = tf.placeholder('float', [batch_size, n_chunks, n_classes])


#roba




def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32) #dynamic

    #flat output

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    # one hot encoding y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

    sess.run(tf.initialize_all_variables())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)): #(bach_size)
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

            # one hot encoding x
            # numpyarray = tensor.eval()

            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))



train_neural_network(x)