"""Train the model
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

from input_data import create_dataset
from input_data import get_iterator_from_datasets
from model.utils import Params
from model.utils import set_logger
from model.model import model
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')


def train(sess, model_spec, model_dir, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for training
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model
        num_steps: (int) train for this number of batches
    """
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    accuracy = model_spec['accuracy']
    sess.run(model_spec['train_init_op'])

    t = trange(num_steps)
    for i in t:
        _, loss_val, accuracy_val = sess.run([train_op, loss, accuracy])
        t.set_postfix(loss='{:05.3f}'.format(loss_val),
                      accuracy='{:05.3f}'.format(accuracy_val))


def train_and_evaluate(model_spec, model_dir, params):
    """Train the model and evalute every epoch.

    Args:
        TODO
    """
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(model_spec['variable_init_op'])

        best_eval_acc = 0.0
        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # Load the training dataset into the pipeline
            sess.run(model_spec['train_init_op'])

            num_steps = (params.train_size + 1) // params.batch_size
            train(sess, model_spec, model_dir, params, num_steps)

            # Save weights
            save_path = os.path.join(model_dir, 'weights', 'after-epoch')
            save_path = saver.save(sess, save_path, global_step=epoch + 1)

            # Load the training dataset into the pipeline
            #sess.run(model_spec['eval_init_op'])

            # Evaluate for one epoch on validation set
            #num_steps = (params.eval_size + 1) // params.batch_size
            #metrics = evaluate(sess, model_spec, model_dir, params, num_steps)

            # If best_eval, best_save_path
            #eval_acc = metrics['accuracy']
            #if eval_acc >= best_eval_acc:
                #best_eval_acc = eval_acc
                #best_save_path = save_path




if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Get the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    mnist = input_data.read_data_sets('data/MNIST', one_hot=False)
    train_images = mnist.train.images
    train_labels = mnist.train.labels.astype(np.int64)
    test_images = mnist.test.images
    test_labels = mnist.test.labels.astype(np.int64)

    # Create the two datasets
    train_dataset = create_dataset(True, train_images, train_labels, params)
    test_dataset = create_dataset(False, test_images, test_labels, params)

    # Create a single iterator and `inputs` dict for the model
    inputs = get_iterator_from_datasets(train_dataset, test_dataset)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model(inputs, 'train', params)

    # TODO: add summaries + tensorboard
    # TODO: add saving and loading in model_dir
    # Train the model
    params.train_size = train_images.shape[0]
    params.eval_size = test_images.shape[0]

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model_spec, args.model_dir, params)
