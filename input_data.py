"""Create the input data pipeline using `tf.data`
"""

import numpy as np
import tensorflow as tf


def input_fn(is_training, images, labels, params):
    """Input function for MNIST.

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        images: (np.ndarray) array of images, with shape (num_samples, 784) and type np.float
        labels: (np.ndarray) array of labels, with shape (num_samples,) and type np.int64
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = images.shape[0]
    assert images.shape[0] == labels.shape[0],\
           "Mismatch between images {} and labels {}".format(images.shape[0], labels.shape[0])

    if is_training:
        # TODO: remove 100
        buffer_size = 100  # whole dataset into the buffer ensures good shuffling
        #buffer_size = num_samples  # whole dataset into the buffer ensures good shuffling
        repeat_count = params.num_epochs
    else:
        buffer_size = 1  # no shuffling
        repeat_count = 1  # only 1 epoch

    # Create a Dataset serving batches of images and labels
    dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(buffer_size=buffer_size)
        #.repeat(params.num_epochs)  TODO: remove
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create an initializable iterator
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.intializer

    images, labels = iterator.get_next()
    inputs = {'images': images, 'labels': labels, 'iterator_init_op': init_op}

    return inputs
