import tensorflow as tf
'''
To find out which devices your operations and tensors are assigned to, 
create the session with log_device_placement configuration option set to True.
'''
# create a graph.
a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3], name='a')
b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3, 2], name='b')
c = tf.matmul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # create a session with log_device_placement set to True.
    # run the ops
    print(sess.run(c))