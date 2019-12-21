"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.prediction import prediction
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'prediction.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_signs")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)
    logging.info('Read {} image filenames for prediction.'.format(params.eval_size))

    # create the iterator over the dataset
    test_inputs = input_fn(tf.estimator.ModeKeys.PREDICT, test_filenames, None, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn(tf.estimator.ModeKeys.PREDICT, test_inputs, params, reuse=False)

    # logging.info("Starting prediction")
    pred = prediction(model_spec, args.model_dir, params, args.restore_from)

    for f, p, l in zip(test_filenames, pred, test_labels):
        if p == l:
            print('{}: Label {} has been predicted with {}... Correct!'.format(f, l, p))
        else:
            print('{}: Label {} has been predicted with {}... Not correct!'.format(f, l, p))
