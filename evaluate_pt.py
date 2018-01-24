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

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test')
parser.add_argument('--restore_file', default='best') # subdir of model_dir with weights


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """Evaluate the model on `num_steps` batches.

    Args:
        model:
        loss_fn:
        data_iterator:
        metrics:
        params:
        num_steps: (int)
    
    """
    
    model.eval()    # set model to eval mode
    summ = []       # summary for eval loop

    # compute metrics over the dataset
    for _ in range(num_steps):
        # prepare the batch by converting numpy arrays to torch Variables
        data_batch, labels_batch = next(data_iterator)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)
        summary_batch = {metric:metrics[metric](output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)
    
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    # Set the random seed for the whole graph
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)        

    # TODO: use case where we just evaluate one model_dir
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data = dataloader.load_data(['test'])
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = dataloader.data_iterator(test_data)

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(restore_file, model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_dir))
    save_dict_to_json(metrics, save_path)
