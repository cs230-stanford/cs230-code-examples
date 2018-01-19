"""Train the model"""

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from input_data import input_fn
from input_data import load_dataset_from_text
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

    # Load Vocabularies
    path_vocab_words = 'data/NER/words.txt'
    path_vocab_tags = 'data/NER/tags.txt'
    vocab_words = tf.contrib.lookup.index_table_from_file(path_vocab_words, num_oov_buckets=1)
    vocab_tags = tf.contrib.lookup.index_table_from_file(path_vocab_tags)
    id_pad_word = vocab_words.lookup(tf.constant('<pad>'))
    id_pad_tag = vocab_tags.lookup(tf.constant('O'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_sentences = load_dataset_from_text('data/NER/test/sentences.txt', vocab_words)
    test_tags = load_dataset_from_text('data/NER/test/tags.txt', vocab_tags)

    # specify the train and eval datasets size
    params.update('data/NER/dataset_params.json')
    params.vocab_size += 1 # to account for unknown words

    # Create iterator over the test set
    inputs = input_fn(False, test_sentences, test_tags, pad_word=id_pad_word,
                            pad_tag=id_pad_tag)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn(False, inputs, params, reuse=False)
    logging.info("- done.")

    logging.info("Starting evaluation")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(tf.tables_initializer())

        # Reload weights from the weights_dir subdirectory
        save_dir = os.path.join(args.model_dir, args.restore_dir)
        save_path = tf.train.latest_checkpoint(save_dir)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.test_size + 1) // params.batch_size
        metrics = evaluate(sess, model_spec, num_steps, None)
        save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_dir))
        save_dict_to_json(metrics, save_path)
