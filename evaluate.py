"""Train the model"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf

from utils.general import Params
from utils.general import set_logger
from utils.tf import evaluate
from model.input import input_fn
from model.input import load_dataset_from_text
from model.model import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/NER',
                    help="Directory containing the dataset")
parser.add_argument('--restore_dir', default='best_weights',
                    help="Subdirectory of model dir containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Get paths for vocabularies and dataset
    path_words = os.path.join(args.data_dir, 'words.txt')
    path_tags = os.path.join(args.data_dir, 'tags.txt')
    path_eval_sentences = os.path.join(args.data_dir, 'dev/sentences.txt')
    path_eval_labels = os.path.join(args.data_dir, 'dev/labels.txt')

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)
    tags = tf.contrib.lookup.index_table_from_file(path_tags)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_sentences = load_dataset_from_text(path_eval_sentences, words)
    test_labels = load_dataset_from_text(path_eval_labels, tags)

    # Specify other parameters for the dataset and the model
    params.eval_size = params.test_size
    params.id_pad_word = words.lookup(tf.constant(params.pad_word))
    params.id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create iterator over the test set
    inputs = input_fn('eval', test_sentences, test_labels, params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_dir)