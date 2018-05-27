import tensorflow as tf
'''
An initializable iterator requires you to run an explicit iterator.
initializer operation before using it. In exchange for this inconvenience, 
it enables you to parameterize the definition of the dataset, 
using one or more tf.placeholder() tensors that can be fed when you initialize the iterator. 
'''
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.contrib.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    # initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        print(value)

    # initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for j in range(100):
        value = sess.run(next_element)
        print(value)
