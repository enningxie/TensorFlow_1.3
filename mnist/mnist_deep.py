import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

FLAGS = None


def conv_op(x, filters, kernel_size=5, strides=1, padding='SAME', activation=True):
    if activation:
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=tf.nn.relu
        )
    else:
        return tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )


def max_pool_op(x, pool_size=2, strides=2, padding='SAME'):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )


def dense_op(x, units, activation=True):
    if activation:
        return tf.layers.dense(
            inputs=x,
            units=units,
            activation=tf.nn.relu
        )
    else:
        return tf.layers.dense(
            inputs=x,
            units=units
        )


def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    with tf.name_scope('conv_1'):
        h_conv1 = conv_op(x_image, 32)
    with tf.name_scope('pool_1'):
        h_pool1 = max_pool_op(h_conv1)
    with tf.name_scope('conv_2'):
        h_conv2 = conv_op(h_pool1, 64)
    with tf.name_scope('pool_2'):
        h_pool2 = max_pool_op(h_conv2)
    with tf.name_scope('fc1'):
        h_flat = tf.contrib.layers.flatten(h_pool2)
        h_fc1 = dense_op(h_flat, 1024)
    with tf.name_scope('dropout'):
        drop_rate = tf.placeholder(tf.float32)
        h_fc1_drop = tf.layers.dropout(h_fc1, rate=drop_rate)
    with tf.name_scope('fc2'):
        y_conv = dense_op(h_fc1_drop, 10, activation=False)
    return y_conv, drop_rate


def main(_):
    # import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # create model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # build the graph.
    y_conv, drop_rate = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy_op, {x: batch[0], y_: batch[1], drop_rate: 0.})
                print('step: {0}, train_accuracy: {1}.'.format(i, train_accuracy))
            sess.run([train_op], {x: batch[0], y_: batch[1], drop_rate: 0.5})

        print('test accuracy: {0}.'.format(accuracy_op.eval({x: mnist.test.images, y_: mnist.test.labels})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/enningxie/Documents/DataSets/mnist')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)