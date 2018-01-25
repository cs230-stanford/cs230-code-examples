import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)
        
    def forward(self, s):
        #                                                  -> batch_size x seq_len
        s = self.embedding(s)                               # batch_size x seq_len x embedding_dim
        s, _ = self.lstm(s)                                 # batch_size x seq_len x lstm_hidden_dim
        s = s.contiguous()                                  # making tensor contiguous in memory
        s = s.view(-1, s.shape[2])                          # batch_size*seq_len x lstm_hidden_dim
        s = self.fc(s)                                      # batch_size*seq_len x num_tags

        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    # labels is has shape batch_size x seq_len
    labels = labels.view(-1)
    mask = (labels>=0).float()  # ignore those which were padded (data_iterator sets them to -1)
    num_tokens = outputs.size()[0]   
    return -torch.sum(outputs[range(num_tokens), labels]*mask)/num_tokens    # TODO: Py3 check!
    
    
def accuracy(outputs, labels):
    # labels in a numpy array with shape batch_size x seq_len
    labels = labels.ravel()                 # this flattens it to batch_size*seq_len
    mask = labels>=0
    outputs = np.argmax(outputs, axis=1)    # outputs are log values! 
    return np.sum(outputs==labels)/float(np.sum(mask))
        
    
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
