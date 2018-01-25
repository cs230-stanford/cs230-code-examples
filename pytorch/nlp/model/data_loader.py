import random
import numpy as np
import json
import os

import torch
from torch.autograd import Variable
import pdb

import utils 

class DataLoader(object):
    def __init__(self, data_dir, params):
        # adding dataset parameters to param (e.g. vocab size, )
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
        self.dataset_params = utils.Params(json_path)        
        
        # loading vocab and inverted vocab (we require this to map words to their indices and vice versa)
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab = {} 
        self.inv_vocab = {}
        with open(vocab_path) as f:
            for i,l in enumerate(f.read().splitlines()):
                self.vocab[l] = i
                self.inv_vocab[i] = l            
        
        # setting the indices for UNKnown words and PADding symbols
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]
                
        # loading tags and inverted tags (we require this to map tags to their indices and vice versa)
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        self.inv_tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i
                self.inv_tag_map[i] = t
        
        # update params reference
        params.update(json_path)
                
        
    def load_sentences_labels(self, sentences_file, labels_file, d):
        sentences = []
        labels = []

        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of UNK_WORD
                s = [self.vocab[token] if token in self.vocab 
                     else self.unk_ind
                     for token in sentence.split(' ')]
                sentences.append(s)
        
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)        
        
        assert len(labels)==len(sentences)
        for i in range(len(labels)):
            assert len(labels[i])==len(sentences[i])
        
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)
    
    
    def load_data(self, types, data_dir):
        data = {}
        print(data_dir)
        train_data_dir = os.path.join(data_dir, "train/")
        val_data_dir = os.path.join(data_dir, "val/")
        test_data_dir = os.path.join(data_dir, "test/")
        
        if 'train' in types:
            train_sentences = os.path.join(train_data_dir, "sentences.txt")
            train_labels = os.path.join(train_data_dir, "labels.txt")
            data['train'] = {}
            self.load_sentences_labels(train_sentences, train_labels, data['train'])
        
        if 'val' in types:
            val_sentences = os.path.join(val_data_dir, "sentences.txt")
            val_labels = os.path.join(val_data_dir, "labels.txt")
            data['val'] = {}
            self.load_sentences_labels(val_sentences, val_labels, data['val'])
        
        if 'test' in types:
            test_sentences = os.path.join(test_data_dir, "sentences.txt")
            test_labels = os.path.join(test_data_dir, "labels.txt")
            data['test'] = {}
            self.load_sentences_labels(test_sentences, test_labels, data['test'])

        return data
        
        
    def data_iterator(self, data, params, shuffle=False):     
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)
    
        for i in range((data['size']+1)//params.batch_size):
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_max_len = max([len(s) for s in batch_sentences])
            
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))          # initialise to -1 to differentiate from padding
            
            for i in range(len(batch_sentences)):
                cur_len = len(batch_sentences[i])
                batch_data[i][:cur_len] = batch_sentences[i]
                batch_labels[i][:cur_len] = batch_tags[i]
            
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    
            yield batch_data, batch_labels