import random
import numpy as np

import torch
from torch.autograd import Variable

from tensorflow.examples.tutorials.mnist import input_data

def load_data(types):
    data = {}
    mnist = input_data.read_data_sets('mnist/data/MNIST', one_hot=False)
    
    if 'train' in types:
        train_images = mnist.train.images[:40000]
        train_labels = mnist.train.labels[:40000].astype(np.int64)
        data['train'] = {}
        data['train']['data'] = train_images
        data['train']['labels'] = train_labels
        data['train']['size'] = train_images.shape[0]
    
    if 'val' in types:
        val_images = mnist.train.images[40000:]
        val_labels = mnist.train.labels[40000:].astype(np.int64)
        data['val'] = {}
        data['val']['data'] = val_images
        data['val']['labels'] = val_labels
        data['val']['size'] = val_images.shape[0]
    
    if 'test' in types:
        test_images = mnist.test.images
        test_labels = mnist.test.labels.astype(np.int64)
        data['test'] = {}
        data['test']['data'] = test_images
        data['test']['labels'] = test_labels
        data['test']['size'] = test_images.shape[0]
    
    return data
    
    
def data_iterator(data, params, shuffle=False):     
    order = list(range(data['size']))
    if shuffle:
        random.seed(230)
        random.shuffle(order)
    
    for i in range((data['size']+1)//params.batch_size):
        batch_data = data['data'][order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_labels = data['labels'][order[i*params.batch_size:(i+1)*params.batch_size]]

        batch_data, batch_labels = torch.from_numpy(batch_data), torch.from_numpy(batch_labels)
        if params.cuda:
            batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
        
        yield batch_data, batch_labels