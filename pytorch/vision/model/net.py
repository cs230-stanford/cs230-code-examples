"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
        # define all the parameters required for the neural network        
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        #                                                  -> batch_size x 3 x 224 x 224
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 224 x 224
        s = F.relu(F.max_pool2d(s,2))                       # batch_size x num_channels x 112 x 112
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 112 x 112
        s = F.relu(F.max_pool2d(s,2))                       # batch_size x num_channels*2 x 56 x 56
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 56 x 56
        s = F.relu(F.max_pool2d(s,2))                       # batch_size x num_channels*4 x 28 x 28
        s = s.view(-1, 8*8*self.num_channels*4)            # batch_size x 7*7*num_channels*16

        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*16
        s = self.fc2(s)                                     # batch_size x 6

        return F.log_softmax(s, dim=1)

def loss_fn(outputs, labels):
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples    # TODO: Py3 check!
    
def accuracy(outputs, labels):
    # outputs are log values!
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)
    
metrics = {
    'accuracy': accuracy,
}
