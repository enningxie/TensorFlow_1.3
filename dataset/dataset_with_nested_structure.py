import tensorflow as tf
'''
If each element of the dataset has a nested structure, 
the return value of Iterator.get_next() will be one or 
more tf.Tensor objects in the same nested structure.
'''

dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.contrib.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
next1, (next2, next3) = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run([next1, next2, next3]))
        except tf.errors.OutOfRangeError:
            print('end of dataset.')
            break