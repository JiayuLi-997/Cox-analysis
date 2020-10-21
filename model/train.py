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
# import distributed.joblib  
from sklearn.externals.joblib import parallel_backend
from collections import Counter
from skorch.callbacks import Checkpoint, PrintLog, EarlyStopping, ProgressBar, TrainEndCheckpoint
import logging
#from data_loader.Dataset import SequenceDataset
from Series_dataset import Series_dataset
from ActionModel import ActionModel
#from models.ActionModel_LSTM import ActionModel_LSTM
#from models.C_ActionModel import C_ActionModel
#from models.U_ActionModel import U_ActionModel
#from models.A_ActionModel import A_ActionModel
#from models.G_ActionModel import G_ActionModel
#from models.CUG_ActionModel import CUG_ActionModel
#from models.UG_ActionModel import UG_ActionModel
#from models.UG_ActionModel_subtract import UG_ActionModel_subtract
#from models.UG_ActionModel_fea import UG_ActionModel_fea
#from models.UG_ActionModel_doc import UG_ActionModel_doc
#from models.UG_ActionModel_dot import UG_ActionModel_dot
#from models.UG_ActionModel_seperate import UG_ActionModel_seperate
#from models.ActionModel_fea import ActionModel_fea
#from models.Conv_ActionModel import Conv_ActionModel
#from models.C_ActionModel_varysize import C_ActionModel_varysize
#from models.C_ActionModel_V2 import C_ActionModel_V2
#from models.ActionModel_Doc import ActionModel_Doc
#from models.ActionModel_UD import ActionModel_UD
#from models.A_UD_ActionModel import A_UD_ActionModel
#from models.UD_rnnModel import UD_RNNModel
#from models.CNN_RNN import CNN_RNN
#from models.UG_CNN import UG_CNN
#from models.Doc_ActionModel import Doc_ActionModel
#from models.ActionGate_DocModel import ActionGate_DocModel
#from models.UG_ActionModel_modify import UG_ActionModel_M
import sys


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('experiment')

logger.info("Experiment start")


def parse_global_args(parser):
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--random_seed', type=int, default=2018,
                        help='Random seed of numpy and tensorflow.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--model_name', type=str, default="GRUModel",
                        help="name of model")
    parser.add_argument('--opt_name', type=str, default="adam",
                        help="name of optimizer")
    parser.add_argument('--loss_name', type=str, default="cross_entropy",
                        help="name of criterion")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2_weight', type=float, default=0.001,
                        help="l2 weight")
    parser.add_argument('--max_epochs', type=int, default=30,
                        help="max epochs")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size")
    parser.add_argument('--device', type=str, default="cuda",
                        help="device")
    parser.add_argument('--record_file', type=str, default="../output/records_new.csv",
                        help="record file")
    parser.add_argument('--W_record_file',type=str,default="../output/W_records.csv",
                        help ="wrong sample record file" )
    parser.add_argument('--cp_path', type=str, default="../output/test", 
                        help="checkpoint file")
    parser.add_argument('--early_stopping', type=int, default=10,
                        help="early stopping epoch")
    parser.add_argument('--attention_l2_weight', type=float, default=0.01, 
                        help="l2 weight for attnetion.attention_weights")
    parser.add_argument('--tr_over_sample', type=int, default=0,
                        help="over sampling when training")
    return parser


def parse_action_args(parser):
    parser.add_argument('--a_embed_dim', type=int, default=64,
                            help='Action embedding dimension')
    parser.add_argument('--a_num_embed', type=int, default=9,
                        help='Number of Actions')
    parser.add_argument('--a_pretrained_embed', type=str, default="",
                        help="Pretrained embeddings for action")
    parser.add_argument('--a_freeze_embed', type=bool, default=False,
                        help="If freeze action embeddings")
    parser.add_argument('--a_padding_idx', type=int, default=0, 
                        help="action padding idx")
    return parser

def parse_rnn_args(parser):
    parser.add_argument('--rnn_hidden_dim', type=int, default=32,
                        help='RNN hidden dimension')
    parser.add_argument('--rnn_bidirectional', type=bool, default=False,
                        help='if rnn bidirectional')
    parser.add_argument('--rnn_num_layers', type=int, default=1,
                        help='RNN number of layers')
    parser.add_argument('--rnn_dropout_p', type=float, default=0.1,
                        help="RNN dropout p")
    return parser

def parse_cnn_args(parser):
    parser.add_argument('--cnn_num_kernels', type=int, default=32,
                        help='CNN number of kernels')
    parser.add_argument('--cnn_kernel_size', type=int, default=6,
                        help='CNN kernel size')
    parser.add_argument('--cnn_stride', type=int, default=1,
                        help='CNN stride')
    return parser

def parse_cate_args(parser):
    parser.add_argument('--c_embed_dim', type=int, default=32,
                            help='Category embedding dimension')
    parser.add_argument('--c_num_embed', type=int, default=64,
                        help='Number of categories')
    parser.add_argument('--c_pretrained_embed', type=str, default="",
                        help="Pretrained embeddings for Category")
    parser.add_argument('--c_freeze_embed', type=bool, default=False,
                        help="If freeze category embeddings")
    parser.add_argument('--c_padding_idx', type=int, default=0, 
                        help="category padding idx")
    return parser

'''
def parse_doc_args(parser):
    parser.add_argument('--d_embed_dim', type=int, default=32,
                            help='Doc embedding dimension')
    parser.add_argument('--d_num_embed', type=int, default=64,
                        help='Number of Docs')
    parser.add_argument('--d_pretrained_embed', type=str, default="",
                        help="Pretrained embeddings for Docs")
    parser.add_argument('--d_freeze_embed', type=bool, default=False,
                        help="If freeze doc embeddings")
    parser.add_argument('--d_padding_idx', type=int, default=0, 
                        help="doc padding idx")
    return parser
'''

def parse_dense_args(parser):
    parser.add_argument('--dropout_p', type=float, default=0.1,
                            help='Dropout')
    parser.add_argument('--dense_hidden_dim', type=int, default=32,
                        help='Dense hidden dimention')
    parser.add_argument('--output_dim', type=int, default=2,
                        help="output dimension")
    return parser

def parse_dataset_args(parser):
    parser.add_argument('--dataset_path', type=str, default="/work/lhy/Data/IJCAI/Dataset/10k/",
                            help='Dataset path')
    parser.add_argument('--input_feature', type=str, default="action_seq,seq_length", help="model input features")
    return parser

def parse_feature_args(parser):
    parser.add_argument('--feature_dim',type=int,default=32,help = "Input feature dimension")
    return parser

"""
build args 
"""
parser = argparse.ArgumentParser()
parser = parse_global_args(parser)
parser = parse_action_args(parser)
parser = parse_rnn_args(parser)
parser = parse_cnn_args(parser)
parser = parse_cate_args(parser)
#parser = parse_doc_args(parser)
parser = parse_dataset_args(parser)
parser = parse_dense_args(parser)
parser = parse_feature_args(parser)

args, extras = parser.parse_known_args()



"""
initialize
"""

session = {}

def record_time(info):
    session[info] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
record_time("time_exp_begin")

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

"""
load pretrained embeddings

"""
if args.c_pretrained_embed != "":
    args.c_pretrained_w2v = pickle.load(open(args.c_pretrained_embed, "rb"))
else:
    args.c_pretrained_w2v = None
    
if args.a_pretrained_embed != "":
    args.a_pretrained_w2v = pickle.load(open(args.a_pretrained_embed, "rb"))
else:
    args.a_pretrained_w2v = None

"""
build model

"""
model_name = eval(args.model_name)

model = model_name(action_embedding_dim=args.a_embed_dim, rnn_hidden_dim=args.rnn_hidden_dim,
                   rnn_num_layers=args.rnn_num_layers, 
                     dense_dropout_p=args.dropout_p, dense_hidden_dim=args.dense_hidden_dim, output_dim=args.output_dim)
    
if args.gpu >= 0:
    model.cuda()

# model.apply(model.init_weights)

"""
optimizer
"""

def _build_optimizer(opt_name):
    logger.info("Optimizer: {}".format(opt_name.title()))
    if opt_name in ['gd', 'adagrad', "adam"]:
        
        param_groups = []
        for key in dict(model.named_parameters()).keys():
            if key == 'attention.attention_weights':
                param_groups.append((key, {"weight_decay": args.attention_l2_weight}))
            else:
                param_groups.append((key, {"weight_decay": args.l2_weight}))
        opt_params = {
            "optimizer__lr" : args.learning_rate,
            "optimizer__weight_decay": args.l2_weight,
            "optimizer__param_groups": param_groups
        }
    else:
        logger.error("Unknown Optimizer: {}".format(opt_name.title()))
        return
    
    if opt_name == "gd":
        optimizer = torch.optim.SGD
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad
    elif opt_name == "adam":
        optimizer = torch.optim.Adam
    
    return optimizer, opt_params

"""
loss
"""

def _build_criterion(loss_name):
    logger.info("Criterion: {}".format(loss_name.title()))
    
    if loss_name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss
    elif loss_name == "nll":
        criterion = torch.nn.NLLLoss
    elif loss_name == "mse":
        criterion = torch.nn.MSELoss
    else:
        logger.error("Unknown Criterion: {}".format(loss_name.title()))
        criterion = torch.nn.CrossEntropyLoss
    return criterion


optimizer, opt_params = _build_optimizer(args.opt_name)
criterion = _build_criterion(args.loss_name)


logger.info("Loading Dataset: {}".format(args.dataset_path))
"""
load data
"""
ds_tr = SequenceDataset.load(os.path.join(args.dataset_path, "train.pkl"))
ds_val = SequenceDataset.load(os.path.join(args.dataset_path, "valid.pkl"))
ds_test = SequenceDataset.load(os.path.join(args.dataset_path, "test.pkl"))

keys = args.input_feature.split(",")
print(keys)

X_tr = ds_tr.to_numpy(keys=[t for t in keys], padding=True, max_length=200)
X_val = ds_val.to_numpy(keys=[t for t in keys], padding=True, max_length=200)
X_test = ds_test.to_numpy(keys=[t for t in keys], padding=True, max_length=200)

# ignore user cates
'''
X_tr = ds_tr.to_numpy(keys=[t for t in keys if t != "user_cates"], padding=True, max_length=200)
X_val = ds_val.to_numpy(keys=[t for t in keys if t != "user_cates"], padding=True, max_length=200)
X_test = ds_test.to_numpy(keys=[t for t in keys if t != "user_cates"], padding=True, max_length=200)

if "user_cates" in keys:
    X_tr['user_cates'] = ds_tr.to_numpy(keys=["user_cates"])["user_cates"]
    X_val['user_cates'] = ds_val.to_numpy(keys=["user_cates"])["user_cates"]
    X_test['user_cates'] = ds_test.to_numpy(keys=["user_cates"])["user_cates"]
'''
y_tr = ds_tr.to_numpy(keys=['stop_seq'], padding=True, max_length=200)['stop_seq']
y_val = ds_val.to_numpy(keys=['stop_seq'], padding=True, max_length=200)['stop_seq']
y_test = ds_test.to_numpy(keys=['stop_seq'], padding=True, max_length=200)['stop_seq']


"""
classifier wrapper
"""
logger.info("Building Classifier: {}".format(args.
                                            ))

cp = Checkpoint(dirname=args.cp_path, monitor="valid_loss_best")
train_end_cp = TrainEndCheckpoint(dirname=args.cp_path, fn_prefix='train_end_')
estop = EarlyStopping(monitor='valid_loss', patience=args.early_stopping, threshold=1e-5, threshold_mode='rel', lower_is_better=True)


classifer = NeuralNetClassifier(
    model,
    
    max_epochs = args.max_epochs,
    lr = args.learning_rate,
    batch_size = args.batch_size,
    criterion = criterion,
    optimizer = optimizer,
    
    device = 'cuda' if args.gpu >= 0 else 'cpu',
    train_split = predefined_split(skDataset(X_val, y_val)),
    **opt_params,
    callbacks=[cp, train_end_cp, estop],
    warm_start = True # Added 2019-01-24
)


"""
training and logging
"""
record_time("time_train_begin")
logger.info("Training...")

if "action_seq" not in X_tr:
    X_tr["action_seq"]=X_tr.pop("doc_seq")
    X_val["action_seq"]=X_val.pop("doc_seq")

classifer.fit(X_tr, y_tr)

record_time("time_train_end")

"""
save
"""
with open(os.path.join(args.cp_path, 'model.pkl'), 'wb') as outf:
    pickle.dump(classifer, outf)
logger.info("Saving to {}".format(os.path.join(args.cp_path, 'model.pkl')))

"""
evaluate and recording
"""
from sklearn import metrics
def score_cls(real, pred_prob=[]):
    #pred = [y_pre>=0.5 for y_pre in pred_prob]+0.0
    print (pred_prob)
    print (type(pred_prob))
    print (len(pred_prob))
    pred = np.array([y_pre>=0.5 for y_pre in pred_prob])+0.0
    #pred = [y+0.0 for y in pred]
    print (type(real))
    print (type(pred))
    print (real.shape)
    print (pred.shape)
    tmat = metrics.confusion_matrix(real, pred)
    neg_p, pos_p = metrics.precision_score(real, pred, average=None)
    neg_r, pos_r = metrics.recall_score(real, pred, average=None)

    f1_neg, f1_pos = metrics.f1_score(real, pred, average=None)
    f1_micro = metrics.f1_score(real, pred, average='micro')
    f1_macro = metrics.f1_score(real, pred, average='macro')
    acc = metrics.accuracy_score(real, pred)
    fpr, tpr, thresholds = metrics.roc_curve(real, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    ycnt = Counter(pred)
    ycnt_value = list(ycnt.values())
    print(ycnt.values())
    return {"Pos P":pos_p, "Pos R":pos_r, "Neg P":neg_p, "Neg R":neg_r, 
            "Pos f1":f1_pos, "Neg f1": f1_neg, "f1_micro": f1_micro, "f1_macro": f1_macro,
            "acc":acc, "auc":auc, 
            "Pred Num": dict(ycnt).items()}

record_time("time_eval_begin")

print("Xval:",type(X_val))
print(len(X_val))
print(X_val['action_seq'].shape)
print("X_test:",type(X_test))
print(len(X_test))
print(X_test['action_seq'].shape)

print("val")
y_val_pred = classifer.predict(X_val)
score = score_cls(y_val, y_val_pred)

print("test")
#y_test_pred = classifer.predict_proba(X_test)
y_test_pred = classifer.predict(X_test)

with open(os.path.join(args.cp_path, 'test_results.pkl'), 'wb') as outf:
    pickle.dump([y_test, y_test_pred], outf)
    
score_test = score_cls(y_test,y_test_pred)

"""
  save wrong samples   
"""
idx_w = (y_val_pred != y_val)
W_data ={}
W_data["action_seq"] = []
[W_data["action_seq"].append(list(X_val["action_seq"][idx,:])) for idx in np.where(idx_w)[0]]
W_data["seq_length"]=X_val["seq_length"][idx_w]
W_data["label"] = y_val[idx_w]
W_data["model"] = args.model_name
W_record = pd.DataFrame(W_data)
W_record.to_csv(args.W_record_file,index = None)                  

record_time("time_eval_end")

def record(cls, score,record_type):
    sess_rec = collections.OrderedDict()
    def add(info, key=None):
        if type(info) == dict:
            for tk in info:
                sess_rec[tk] = info[tk]
        else:
            sess_rec[key] = info
    sess_rec["record_type"]=record_type
    """add time"""
    add(session["time_exp_begin"], "time_exp_begin")
    add(session['time_train_begin'], "time_train_begin")
    add(session['time_train_end'], "time_train_end")
    add(session['time_eval_begin'], "time_eval_begin")
    add(session['time_eval_end'], "time_eval_end")

    """add parameter"""
    if 'a_pretrained_w2v' in vars(args):
        vars(args).pop('a_pretrained_w2v')
        vars(args).pop('c_pretrained_w2v')
    add(vars(args))
    #add(args_dict)
    """add score"""
    add(score)
    
    columns = ['model_name',
                 'Neg P',
                 'Neg R',
                 'Neg f1',
                 'Pos P',
                 'Pos R',
                 'Pos f1',
                 'acc',
                 'auc',
                 'Pred Num',
                 "f1_micro", 
                 "f1_macro",
                 "tr_over_sample",
                 'a_embed_dim',
                 'a_freeze_embed',
                 'a_num_embed',
                 'a_padding_idx',
                 'a_pretrained_embed',
                 'rnn_bidirectional',
                 'rnn_hidden_dim',
                 'rnn_num_layers',
                 'c_embed_dim',
                 'c_freeze_embed',
                 'c_num_embed',
                 'c_padding_idx',
                 'c_pretrained_embed',
                 'cnn_kernel_size',
                 'cnn_num_kernels',
                 'cnn_stride',
                 'l2_weight',
                 'attention_l2_weight',
                 'learning_rate',
                 'loss_name',
                 'max_epochs',
                 'opt_name',
                 'dropout_p',
                 'early_stopping',
                 'batch_size',
                 'dense_hidden_dim',
                 'random_seed',
                 'cp_path',
                 'dataset_path',
                 'record_file',
                 'time_exp_begin',
                 'time_train_begin',
                 'time_train_end',
                 'time_eval_begin',
                 'time_eval_end',
                 'record_type',
                 'feature_dim']
                 
    df_sess_rec = pd.DataFrame(sess_rec, index=[0])
    
    if not os.path.exists(args.record_file):
        df_record = df_sess_rec[columns]
    else:
        df_record = pd.read_csv(args.record_file)
        for column in columns:
            if column not in df_record.columns.values.tolist():
                df_record[column]=pd.Series([])
        df_record= df_record[columns].append(df_sess_rec[columns], sort=False)
        #df_record[columns].append(df_sess_rec[columns], sort=False)
        df_record.reset_index(drop=True, inplace=True)
    df_record.to_csv(args.record_file, index=None)
    return df_record

df_record = record(classifer, score,'val')
df_record_test = record(classifer, score_test,'test')
print (df_record.iloc[-1,:])
print (df_record_test.iloc[-1,:])

