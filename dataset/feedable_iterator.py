import tensorflow as tf
'''
A feedable iterator can be used together with tf.placeholder to select what Iterator 
to use in each call to tf.Session.run, via the familiar feed_dict mechanism. 
It offers the same functionality as a reinitializable iterator, but it does not 
require you to initialize the iterator from the start of a dataset when you switch 
between iterators. 
'''

# define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
).repeat()
validation_dataset = tf.contrib.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes
)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

with tf.Session() as sess:
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            print(sess.run(next_element, feed_dict={handle: training_handle}))

        # run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            print(sess.run(next_element, feed_dict={handle: validation_handle}))

