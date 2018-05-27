import tensorflow as tf
'''
If you would like a particular operation to run on a device of your choice instead of 
what's automatically selected for you, you can use with tf.device to create a device 
context such that all the operations within that context will have the same device assignment.
'''

# create a graph.
with tf.device('/cpu:0'):
    a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3], name='a')
    b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3, 2], name='b')

c = tf.matmul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # run the ops.
    print(sess.run(c))