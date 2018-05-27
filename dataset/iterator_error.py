import tensorflow as tf
'''
If the iterator reaches the end of the dataset, executing the Iterator.get_next() operation 
will raise a tf.errors.OutOfRangeError. After this point the iterator will be in an unusable 
state, and you must initialize it again if you want to use it further.
'''

dataset = tf.contrib.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

result = tf.add(next_element, next_element)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(result))  # ===> '0'
    print(sess.run(result))  # ===> '2'
    print(sess.run(result))  # ===> '4'
    print(sess.run(result))  # ===> '6'
    print(sess.run(result))  # ===> '8'
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        print('end of dataset.')

# A common pattern is to wrap the "training loop" in a try-except block.
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(result))
        except tf.errors.OutOfRangeError:
            print('end of dataset again.')
            break