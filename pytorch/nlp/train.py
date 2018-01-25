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
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small')
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_file', default=None)


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model:
        optimizer: 
        loss_fn: 
        data_iterator: 
        metrics: 
        params:
        num_steps: (int) number of batches to train on, each of size params.batch_size
        summary:
    """

    model.train()   # set model to training mode
    summ = []       # summary for current training loop
    loss_avg = utils.RunningAverage()   # an object that maintains the running average
    
    # Use tqdm for progress bar
    t = trange(num_steps) 
    for i in t:
        # prepare the batch by converting numpy arrays to torch Variables
        train_batch, labels_batch = next(data_iterator)
                    
        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        
        # clear gradients, compute new gradients and perform update
        optimizer.zero_grad()
        loss.backward()  # computes gradients of all variables wrt loss
        optimizer.step()

        # Evaluate summaries only once in a while        
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        if i % params.save_summary_steps == 0:
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)
        
        loss_avg.update(loss.data[0])
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    

def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model:
        train_data:
        optimizer: 
        loss_fn: 
        metrics: 
        params:
        model_dir: 
        restore_file:
    """
    # reload weights from directory if specified
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        utils.load_checkpoint(restore_file, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)
            
        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
        
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
    

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()     # use GPU is available
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
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
