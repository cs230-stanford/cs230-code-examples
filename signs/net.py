import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        
        # define all the parameters required for the neural network        
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)

        self.fc1 = nn.Linear(7*7*64, 128)        
        self.fc2 = nn.Linear(128, 10)        
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        #                                                  -> batch_size x 784
        s = s.view(-1, 1, 28, 28)                           # batch_size x 1 x 28 x 28
        s = self.conv1(s)                                   # batch_size x 64 x 28 x 28
        s = F.relu(F.max_pool2d(s,2))                       # batch_size x 64 x 14 x 14
        s = self.conv2(s)                                   # batch_size x 64 x 14 x 14
        s = F.relu(F.max_pool2d(s,2))                       # batch_size x 64 x 7 x 7
        s = s.view(-1, 7*7*64)                              # batch_size x 7*7*64

        s = F.dropout(F.relu(self.fc1(s)), 
            p=self.dropout_rate, training=self.training)    # batch_size x 128
        s = self.fc2(s)                                     # batch_size x 10

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
