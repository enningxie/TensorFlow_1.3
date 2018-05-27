import tensorflow as tf
'''
A one-shot iterator is the simplest form of iterator, which only supports iterating once 
through a dataset, with no need for explicit initialization. One-shot iterators handle 
almost all of the cases that the existing queue-based input pipelines support, but 
they do not support parameterization.
'''

dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        print(value)