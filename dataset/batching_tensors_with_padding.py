import tensorflow as tf
'''
However, many models (e.g. sequence models) work with input data that can have varying size 
(e.g. sequences of different lengths). To handle this case, the Dataset.padded_batch() 
transformation enables you to batch tensors of different shape by specifying one or more 
dimensions in which they may be padded.
'''

dataset = tf.contrib.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))
    print(sess.run(next_element))