import tensorflow as tf


# read an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


# A vector of filenames.
filenames = tf.constant(['/var/data/image1.jpg', '/var/data/image2.jpg', ...])

# labels[i] is the label for the image in filenames[i]
labels = tf.constant([0, 37, ...])

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

#############################other ops about dataset.###############################################
# map shuffle batch repeat.
dataset = dataset.map(...)  # parse the record into tensors.
dataset = dataset.shuffle(buffer_size=10000)  # shuffle_op.
dataset = dataset.batch(32)  # batch_op
dataset = dataset.repeat()  # repeat the input indefinitely.
