"""Train the model"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from input_data import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_dir', default='best_weights') # subdir of model_dir with weights


def evaluate(sess, model_spec, num_steps, writer=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # TODO: use case where we just evaluate one model_dir
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = "data/SIGNS"
    test_data_dir = os.path.join(data_dir, "test_signs")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames]

    # specify the test set size
    test_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, params)

    # Define the model
    logging.info("Creating the model...")
    test_model_spec = model_fn(False, test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Reload weights from the weights_dir subdirectory
        save_dir = os.path.join(args.model_dir, args.restore_dir)
        save_path = tf.train.latest_checkpoint(save_dir)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (test_size + 1) // params.batch_size
        metrics = evaluate(sess, test_model_spec, num_steps, None)
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_dir))
        save_dict_to_json(metrics, save_path)
