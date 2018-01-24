"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_data import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


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
    data_dir = args.data_dir
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
    test_model_spec = model_fn('eval', test_inputs, params, reuse=False)

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
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_from))
        save_dict_to_json(metrics, save_path)
