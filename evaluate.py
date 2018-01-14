"""Train the model
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

from input_data import input_fn
from model.utils import Params
from model.utils import set_logger
from model.model import model
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')


def evaluate(model_spec, model_dir, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for training
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model
        num_steps: (int) train for this number of batches
    """
    sess.run(model_spec['iterator_init_op'])

        t = trange(num_steps)
        for i in t:
            _, loss_val, accuracy_val = sess.run([train_op, loss, accuracy])
            t.set_postfix(loss='{:05.3f}'.format(loss_val),
                          accuracy='{:06.2f}'.format(accuracy_val * 100))


if __name__ == '__main__':
    # TODO: use case where we just evaluate one model_dir
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Get the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    mnist = input_data.read_data_sets('data/MNIST', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels.astype(np.int64)
    inputs = input_fn(False, test_images, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model(inputs, 'eval', params)

    # TODO: add summaries + tensorboard
    # TODO: add saving and loading in model_dir
    # Test the model
    params.test_size = test_images.shape[0]

    logging.info("Starting evaluation")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Reload weights
        save_path = os.path.join(model_dir, 'best')  # TODO: get correct path
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.test_size + 1) // params.batch_size
        metrics = evaluate(sess, model_spec, args.model_dir, params)

        for key in metrics:
            tf.logging.info("{}: {}".format(key, metrics[key]))
