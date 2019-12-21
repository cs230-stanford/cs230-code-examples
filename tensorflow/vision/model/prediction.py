"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json


def predict_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    # Load the prediction dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])

    # compute predictions over the dataset
    all_pred = []
    for _ in range(num_steps):
        all_pred.extend(sess.run(model_spec['predictions']))

    return all_pred


def prediction(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for prediction
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        pred = predict_sess(sess, model_spec, num_steps)

    return pred
