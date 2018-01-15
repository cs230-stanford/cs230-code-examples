"""Train the model
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from input_data import create_dataset
from input_data import get_iterator_from_dataset
from model.utils import Params
from model.utils import set_logger
from model.model import model


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')


def evaluate(sess, writer, model_spec, num_steps):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        writer: (tf.summary.FileWriter) writer for summaries
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['eval_metrics']
    global_step = tf.train.get_global_step()

    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    global_step_val = sess.run(global_step)
    for tag, val in metrics_val.items():
        summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
        writer.add_summary(summ, global_step_val)

    return metrics_val


if __name__ == '__main__':
    tf.set_random_seed(230)

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
    test_dataset = create_dataset(False, test_images, test_labels, params)
    inputs = get_iterator_from_dataset(test_dataset)

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
        metrics = evaluate(sess, model_spec, num_steps)

        for key in metrics:
            tf.logging.info("{}: {}".format(key, metrics[key]))
