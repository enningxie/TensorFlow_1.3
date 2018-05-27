import tensorflow as tf

# create some variables.
v1 = tf.get_variable('v1', [3], initializer=tf.zeros_initializer)
v2 = tf.get_variable('v2', [5], initializer=tf.zeros_initializer)

# add ops to save and restore only 'v2' using the name 'v2'.
saver = tf.train.Saver({'v2': v2})

with tf.Session() as sess:
    v1.initializer.run()
    saver.restore(sess, '/home/enningxie/tmp/model/test.ckpt')
    print('v1: {0}'.format(v1.eval()))
    print('v2: {0}'.format(v2.eval()))
    tf.ins