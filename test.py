import tensorflow as tf

a = tf.Variable([1., 2., 3.], dtype=tf.float32)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init_op)
    print(sess.run(tf.report_uninitialized_variables()))