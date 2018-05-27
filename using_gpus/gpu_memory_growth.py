import tensorflow as tf

# The first is the allow_growth option, which attempts to allocate only as much GPU memory
# based on runtime allocations: it starts out allocating very little memory, and as Sessions
# get run and more GPU memory is needed, we extend the GPU memory region needed by the
# TensorFlow process. Note that we do not release memory, since that can lead to even
# worse memory fragmentation. To turn this option on, set the option in the ConfigProto.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# The second method is the per_process_gpu_memory_fraction option, which determines the
# fraction of the overall amount of memory that each visible GPU should be allocated.
# For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU.

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)