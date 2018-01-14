"""Train the model
"""

import argparse
import os

import tensorflow as tf
from tqdm import tqdm

from input_data import input_fn
from model.utils import Params
from model.model import model


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')


def train(model_spec, model_dir, num_steps):
    """Train the model
    """
    accuracy = model_spec['accuracy']
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    init_op = model_spec['init_op']

    with tf.Session() as sess:
        sess.run(init_op)
        for t in tqdm(range(num_steps)):
            _, loss_val, accuracy_val = sess.run([train_op, loss, accuracy])
            print(loss_val, accuracy_val)


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path)
    params = Params(json_path)

    # Create the input data pipeline
    inputs = input_fn(True, params)

    # Define the model
    model_spec = model(inputs, 'train', params)

    # Train the model
    train_size = 55000
    num_steps = (train_size * params.num_epochs + 1) // params.batch_size
    train(model_spec, args.model_dir, num_steps)
