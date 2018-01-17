"""Create the input data pipeline using `tf.data`
"""

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
        buffer_size = num_samples  # whole dataset into the buffer ensures good shuffling
    else:
        buffer_size = 1  # no shuffling

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(buffer_size=buffer_size)
        .batch(params.batch_size)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': init_op}
    return inputs
