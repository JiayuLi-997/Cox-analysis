import os
from argparse import Namespace
import collections
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset as skDataset
from gensim.models import Word2Vec
import pickle
# import distributed.joblib
from sklearn.externals.joblib import parallel_backend
from collections import Counter
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.autograd import Variable


class GRUModel(nn.Module):
    def __init__(self,
                 action_embedding_dim, #action_num_embeddings, action_pretrained_embeddings=None, 
                 #action_freeze_embeddings=False, action_padding_idx=0,
                 rnn_hidden_dim=32, rnn_bidirectional=False, rnn_num_layers=1, 
                 dense_dropout_p=0.1, dense_hidden_dim=32, output_dim=2,out_hidden_dim=16):
        
        super(GRUModel, self).__init__()
        
        # Embedding
        #if action_pretrained_embeddings is not None:
        #    action_pretrained_embeddings = torch.from_numpy(action_pretrained_embeddings).float()
        #self.action_embeddings = nn.Embedding(embedding_dim=action_embedding_dim,
        #                               num_embeddings=action_num_embeddings,
        #                               padding_idx=action_padding_idx,
        #                               _weight=action_pretrained_embeddings)
       
        # Gru
        self.gru = nn.GRU(input_size=action_embedding_dim, hidden_size=rnn_hidden_dim, 
                          num_layers=rnn_num_layers, batch_first=True, bidirectional=rnn_bidirectional)
     
        # FC weights
        self.dropout = nn.Dropout(dense_dropout_p)
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)
        
        #if action_freeze_embeddings:
        #    self.action_embeddings.weight.requires_grad = False

    def forward(self, x, seq_length):        
        #x = self.action_embeddings(action_seq)
        
        out, h_n = self.gru(x) # h_n is the last hidden state (same as out[-1])
        out = self.__gather_last_output(out, seq_length)

        z = self.dropout(out)
        y_pred = self.fc(z)
        return y_pred
    
    def __gather_last_output(self, output, seq_length):
        seq_length = seq_length.long().detach().cpu().numpy() - 1
        out = []
        for batch_index, column_index in enumerate(seq_length):
            out.append(output[batch_index, column_index])
        return torch.stack(out)
 