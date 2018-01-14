"""Create the input data pipeline using `tf.data`
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# TODO: add is_training argument in code
def input_fn(is_training, params):
    """Input function for MNIST
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    train_images = mnist.train.images
    train_labels = mnist.train.labels.astype(np.int64)
    train_size = train_images.shape[0]

    # TODO: check if training or not
    # TODO: document more each line
    dataset = (tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(buffer_size=train_size)
        .repeat(params.num_epochs)
        .batch(params.batch_size)
        .prefetch(1)
    )

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    inputs = {'images': images, 'labels': labels}
    return inputs
