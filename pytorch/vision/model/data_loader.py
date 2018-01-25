import random
import numpy as np
import os

from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import pdb


# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
loader = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()])  # transform it into a torch tensor    
    
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
    
    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))
            filenames = os.listdir(path)
            filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]
            images = load_set(filenames)
            labels = [int(filename.split('/')[-1][0]) for filename in filenames]
            data[split] = {}
            data[split]['data'] = images
            data[split]['labels'] = torch.LongTensor(labels)
            data[split]['size'] = images.shape[0]
    
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