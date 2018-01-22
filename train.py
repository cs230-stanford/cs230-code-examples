"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from utils.general import Params
from utils.general import set_logger
from utils.tf import train_and_evaluate

from model.input import input_fn
from model.input import load_dataset_from_text
from model.model import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_dir', default=None)


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(params.path_words, num_oov_buckets=1)
    tags = tf.contrib.lookup.index_table_from_file(params.path_tags)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_sentences = load_dataset_from_text(params.train_sentences, words)
    train_labels = load_dataset_from_text(params.train_labels, tags)
    test_sentences = load_dataset_from_text(params.test_sentences, words)
    test_labels = load_dataset_from_text(params.test_labels, tags)

    # specify the train and eval datasets size
    params.update(params.dataset_params)
    params.eval_size = params.test_size
    params.vocab_size += 1 # to account for unknown words
    id_pad_word = words.lookup(tf.constant(params.pad_word))
    id_pad_tag = tags.lookup(tf.constant(params.pad_tag))

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_sentences, train_labels, pad_word=id_pad_word,
                            pad_tag=id_pad_tag)
    eval_inputs = input_fn(False, test_sentences, test_labels, pad_word=id_pad_word,
                            pad_tag=id_pad_tag)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn(True, train_inputs, params)
    eval_model_spec = model_fn(False, eval_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_dir)
