import random
import numpy as np
import os

from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import pdb
# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
loader = transforms.ToTensor()  # transform it into a torch tensor    
    
def image_loader(filename):
    image = Image.open(filename)
    image = loader(image)
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image
    
def load_set(filenames):
    images = []
    for filename in filenames:
        images.append(image_loader(filename))

    images = torch.cat(images)
    return images
        
def load_data(types, data_dir):
    data = {}
    
    train_data_dir = os.path.join(data_dir, "train_signs")
    val_data_dir = os.path.join(data_dir, "val_signs")
    test_data_dir = os.path.join(data_dir, "test_signs")
    
    if 'train' in types:
        train_filenames = os.listdir(train_data_dir)
        train_filenames = [os.path.join(train_data_dir, f) for f in train_filenames if 'DS_Store' not in f]
        train_images = load_set(train_filenames)
        train_labels = [int(filename.split('/')[-1][0]) for filename in train_filenames]
        data['train'] = {}
        data['train']['data'] = train_images
        data['train']['labels'] = torch.from_numpy(np.array(train_labels))
        data['train']['size'] = train_images.shape[0]
    
    if 'val' in types:
        val_filenames = os.listdir(val_data_dir)
        val_filenames = [os.path.join(val_data_dir, f) for f in val_filenames if 'DS_Store' not in f]
        val_images = load_set(val_filenames)
        val_labels = [int(filename.split('/')[-1][0]) for filename in val_filenames]
        data['val'] = {}
        data['val']['data'] = val_images
        data['val']['labels'] = torch.from_numpy(np.array(val_labels))
        data['val']['size'] = val_images.shape[0]
    
    if 'test' in types:
        test_filenames = os.listdir(test_data_dir)
        test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if 'DS_Store' not in f]
        test_images = load_set(test_filenames)
        test_labels = [int(filename.split('/')[-1][0]) for filename in test_filenames]
        data['test'] = {}
        data['test']['data'] = test_images
        data['test']['labels'] = torch.from_numpy(np.array(test_labels))
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
        
        if params.cuda:
            batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        yield batch_data, batch_labels