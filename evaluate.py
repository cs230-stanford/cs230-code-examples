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
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_dir', default='best_weights') # subdir of model_dir with weights


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(params.path_words, num_oov_buckets=1)
    tags = tf.contrib.lookup.index_table_from_file(params.path_tags)

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_sentences = load_dataset_from_text(params.test_sentences, words)
    test_labels = load_dataset_from_text(params.test_labels, tags)

    # Specify the train and eval datasets size
    params.update(params.dataset_params)
    params.vocab_size += 1 # to account for unknown words
    id_pad_word = words.lookup(tf.constant(params.pad_word))
    id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create iterator over the test set
    inputs = input_fn(False, test_sentences, test_labels, pad_word=id_pad_word,
                            pad_tag=id_pad_tag)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn(False, inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_dir)