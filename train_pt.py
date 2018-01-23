"""Train the model"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

import utils
import mnist.net as net
import mnist.dataloader as dataloader
from evaluate_pt import evaluate
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_file', default=None)


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps, summary):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        params: (Params) hyperparameters
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
    """

    model.train()   # set model to training mode
    summ = []       # summary for current training loop
    
    # Use tqdm for progress bar
    t = trange(num_steps) 
    for i in t:
        # prepare the batch by converting numpy arrays to torch Variables
        train_batch, labels_batch = next(data_iterator)
        train_batch, labels_batch = torch.from_numpy(train_batch), torch.from_numpy(labels_batch)
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
                    
        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        
        # clear gradients, compute new gradients and perform update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate summaries only once in a while        
        if i % params.save_summary_steps == 0:
            summary_batch = {metric:metrics[metric](output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

        t.set_postfix(loss='{:05.3f}'.format(loss.data[0]))

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    summary.append(summ)
    

def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        val_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, val_size, save_summary_steps
    """
    # reload weights from directory if specified
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        utils.load_checkpoint(restore_file, model, optimizer)

    best_val_acc = 0.0
    train_summary = []

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = dataloader.data_iterator(train_data, params.batch_size, shuffle=False)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps, train_summary)
            
        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = dataloader.data_iterator(val_data, params.batch_size, shuffle=False)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, params, metrics, num_steps)
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    
    # take care of train_summary

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # load data
    data = dataloader.load_data(['train', 'val'])
    train_data = data['train']
    val_data = data['val']

    # specify the train and val datasets size
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)
