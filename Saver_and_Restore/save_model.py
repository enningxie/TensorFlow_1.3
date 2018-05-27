import tensorflow as tf

# create some variables.
v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(inc_v1)
    sess.run(dec_v2)
    print(sess.run([v1, v2]))

    # define the path.
    save_path = saver.save(sess, '/home/enningxie/tmp/model/test.ckpt')
    print('model saved in file {0}.'.format(save_path))