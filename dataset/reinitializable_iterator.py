import tensorflow as tf
'''
A reinitializable iterator can be initialized from multiple different Dataset objects. 
For example, you might have a training input pipeline that uses random perturbations 
to the input images to improve generalization, and a validation input pipeline that 
evaluates predictions on unmodified data. These pipelines will typically use different 
Dataset objects that have the same structure (i.e. the same types and compatible shapes 
for each component).
'''

# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
)
validation_dataset = tf.contrib.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
        # initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            print(sess.run(next_element))
        # initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            print(sess.run(next_element))

