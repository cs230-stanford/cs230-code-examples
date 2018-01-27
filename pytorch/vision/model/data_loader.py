import random
import os

from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms


# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# define an image loader that specifies transforms on images. See documentation for more details.
loader = transforms.Compose([
    transforms.Resize(64),   # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor    


def image_loader(filename):
    """
    Loads image from filename.

    Args:
        filename: (string) path of image to be loaded

    Returns:
        image: (Tensor) contains data of the image
    """
    image = Image.open(filename)    # PIL image
    image = loader(image)           # dim: 3 x 64 x 64 (applies transformations from loader)
    image = image.unsqueeze(0)      # dim: 1 x 3 x 64 x 64 (fake batch dimension added)
    return image


def load_set(filenames):
    """
    Load all images in filenames.

    Args:
        filenames: (list) contains all filenames from which images are to be loaded.

    Returns:
        images: (Tensor) contains all image data
    """
    images = []
    for filename in filenames:
        images.append(image_loader(filename))

    # images is a list where each image is a Tensor with dim 1 x 3 x 64 x 64
    # we concatenate them into one Tensor of dim len(images) x 3 x 64 x 64
    images = torch.cat(images)
    return images


def load_data(types, data_dir):
    """
    Loads the data for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset

    Returns:
        data: (dict) contains the data with labels for each type in types
    """
    data = {}    
    
    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))
            filenames = os.listdir(path)
            filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]

            # load the images from the corresponding files
            images = load_set(filenames)

            # labels are present in the filename itself for SIGNS
            labels = [int(filename.split('/')[-1][0]) for filename in filenames]

            # storing images and labels in dict
            data[split] = {}
            data[split]['data'] = images
            # since labels are indices, we convert them to torch LongTensors
            data[split]['labels'] = torch.LongTensor(labels)
            data[split]['size'] = images.shape[0]
    
    return data
    
    
def data_iterator(data, params, shuffle=False):
    """
    Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
    pass over the data.

    Args:
        data: (dict) contains data which has keys 'data', 'labels' and 'size'
        params: (Params) hyperparameters of the training process.
        shuffle: (bool) whether the data should be shuffled

    Yields:
        batch_data: (Variable) dimension batch_size x 3 x 64 x 64 with the sentence data
        batch_labels: (Variable) dimension batch_size with the corresponding labels
    """

    # make a list that decides the order in which we go over the data- this avoids explicit shuffling of dat
    order = list(range(data['size']))
    if shuffle:
        random.seed(230)
        random.shuffle(order)

    # one pass over data
    for i in range((data['size']+1)//params.batch_size):
        # fetch images and labels
        batch_data = data['data'][order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_labels = data['labels'][order[i*params.batch_size:(i+1)*params.batch_size]]

        # shift tensors to GPU if available
        if params.cuda:
            batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

        # convert them to Variables to record operations in the computational graph
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        yield batch_data, batch_labels
