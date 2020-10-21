# coding: utf-8

import os
import collections
import argparse
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import random
import time
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
#import distributed.joblib  
from sklearn.externals.joblib import parallel_backend
from collections import Counter
from skorch.callbacks import Checkpoint, PrintLog, EarlyStopping, ProgressBar, TrainEndCheckpoint
import logging
from data_loader.Dataset import SequenceDataset
import sklearn.metrics as metrics
import scipy


def parse_global_args(parser):
    parser.add_argument('--model_path', type=str, default='../output/test/',
                        help='model file path')
    parser.add_argument('--dataset_path', type=str, default='/work/luhongyu/Data/IJCAI/Dataset/10k/',
                        help="dataset path")
    parser.add_argument('--random_seed', type=int, default=2018,
                        help="random seed")
    parser.add_argument('--num_parts', type=int, default=10,
                        help="number of parts in evaluation")
    parser.add_argument('--padding', type=bool, default=True,
                        help="if padding")
    parser.add_argument('--max_length', type=int, default=200,
                        help="max length")
    parser.add_argument('--input_feature', type=str, default="action_seq,seq_length", 
                        help="model input features")
    return parser


def score_cls(real, pred=[]):
    tmat = metrics.confusion_matrix(real, pred)
    Nlv_P,Leave_P = metrics.precision_score(real, pred, average=None)
    Nlv_R,Leave_R = metrics.recall_score(real, pred, average=None)
    
    f1_neg, f1_pos = metrics.f1_score(real, pred, average=None)
    acc = metrics.accuracy_score(real, pred)
    fpr, tpr, thresholds = metrics.roc_curve(real, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return {"Pos P":Leave_P,"Pos R":Leave_R,"Neg P":Nlv_P,"Neg R":Nlv_R, "Pos f1":f1_pos, "Neg f1": f1_neg, "acc":acc, "auc":auc}

def evaluate_partbypart(model, X, y, length, num_parts = 10):
    random.seed(args.random_seed)

    def get_part_data(X, y, idx):
        ansx = {}
        for tk in X:
            ansx[tk] = X[tk][idx]
        ansy = y[idx]
        return ansx, ansy
    
    part_len = length // num_parts
    
    recs = []
    idxs = np.arange(length)
    random.shuffle(idxs)
    
    for i in range(num_parts):
        px, py = get_part_data(X, y, idxs[part_len * i : part_len * (i + 1)])
        py_pred = model.predict(px)
        sc = score_cls(py, py_pred)
        recs.append(sc)
    
    df_ans = pd.DataFrame.from_dict(recs)
    return df_ans


"""
load args
"""
parser = argparse.ArgumentParser()
parser = parse_global_args(parser)
args, extras = parser.parse_known_args()

"""
load model
"""
with open(os.path.join(args.model_path, "model.pkl"), 'rb') as inf:
    classifier = pickle.load(inf)


"""
load data
"""
keys = args.input_feature.split(",")
ds_te = SequenceDataset.load(os.path.join(args.dataset_path, "test.pkl"))
#X_te = ds_te.to_numpy(keys=['action_seq', "seq_length"], padding=args.padding, max_length=args.max_length)
X_te = ds_te.to_numpy(keys=[t for t in keys], padding=args.padding, max_length=args.max_length)
y_te = ds_te.to_numpy(keys=['stop_seq'], padding=args.padding, max_length=args.max_length)['stop_seq']


"""
evaluate with test set
"""
df_single = evaluate_partbypart(classifier, X_te, y_te, length=ds_te.length, num_parts=1)
df_part = evaluate_partbypart(classifier, X_te, y_te, length=ds_te.length, num_parts=args.num_parts)

df_ans = pd.concat([df_single, df_part])
df_ans.index = ['all'] + list(range(args.num_parts))


"""
save result
"""
df_ans.to_csv(os.path.join(args.model_path, "test_result.csv"))
print (df_ans.round(4))

y_te_pred = classifier.predict(X_te)

with open(os.path.join(args.model_path, 'test_results.pkl'), 'wb') as outf:
    pickle.dump([y_te, y_te_pred], outf)
