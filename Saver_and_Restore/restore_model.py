import tensorflow as tf

v1 = tf.get_variable('v1', shape=[3])
v2 = tf.get_variable('v2', shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    # restore variables from disk.
    saver.restore(sess, '/home/enningxie/tmp/model/test.ckpt')
    print('Model restored.')
    # check
    print(sess.run([v1, v2]))