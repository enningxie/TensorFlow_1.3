import tensorflow as tf
import time
# create a graph.
c = []
for d in ['/cpu:0', '/gpu:0']:
    with tf.device(d):
        start_time = time.time()
        a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3])
        b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3, 2])
        c.append(tf.matmul(a, b))
        print('cost time: {0}.'.format(time.time() - start_time))

with tf.device('/gpu:0'):
    sum = tf.add_n(c)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum))