#!/usr/bin/env python

"""
    bow.py
"""

from __future__ import print_function, division

import os
import re
import string
import numpy as np
from tqdm import tqdm

from rsub import *
from matplotlib import pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch import autograd
from torch import nn
from torch.utils.data import dataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler

from collections import defaultdict
from helpers import to_numpy, set_seeds
from hyperlayer import HyperLayer

# --
# Helpers

def texts_from_folders(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(open(fpath).read())
            labels.append(idx)
    return texts,np.array(labels)


def bow2adjlist(X, maxcols=None):
    x = coo_matrix(X)
    _, counts = np.unique(x.row, return_counts=True)
    pos = np.hstack([np.arange(c) for c in counts])
    adjlist = csr_matrix((x.col + 1, (x.row, pos)))
    datlist = csr_matrix((x.data, (x.row, pos)))
    
    if maxcols is not None:
        adjlist, datlist = adjlist[:,:maxcols], datlist[:,:maxcols]
    
    return adjlist, datlist


def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def calc_r(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))


# --
# IO

names = ['neg', 'pos']
_, y_train = texts_from_folders('/home/bjohnson/software/fastai/courses/dl1/data/aclImdb/train', names)
_, y_test = texts_from_folders('/home/bjohnson/software/fastai/courses/dl1/data/aclImdb/test', names)

# --
# Preprocess

X_train = np.load('X_train.npy').item()
X_test = np.load('X_test.npy').item()

X_train_words = np.load('X_train_words.npy').item()
X_test_words = np.load('X_test_words.npy').item()

vocab_size = X_train.shape[1]
n_classes = int(y_train.max()) + 1

X_train_ = Variable(torch.from_numpy(X_train_words.toarray()).long())
y_train_ = Variable(torch.from_numpy(y_train))

X_test_ = Variable(torch.from_numpy(X_test_words.toarray()).long())
y_test_ = Variable(torch.from_numpy(y_test))

X_train_, y_train_, X_test_, y_test_ = X_train_.cuda(), y_train_.cuda(), X_test_.cuda(), y_test_.cuda()


# --
# Define model

from torch.nn import Parameter
class DotProdNB(nn.Module):
    def __init__(self, vocab_size, r_emb, alpha):
        
        super(DotProdNB, self).__init__()
        
        # Init w
        self.w = nn.Embedding(vocab_size + 1, 1, padding_idx=0).cuda()
        self.w.weight.data.uniform_(-0.1, 0.1)
        
        self.fc = nn.Linear(2, 2, bias=False).cuda()
        self.fc.weight.data.set_(torch.eye(2).cuda())
        
        # Init r
        self.r = r_emb
        self.r.weight.requires_grad = False
        
        self.r_noise = nn.Embedding(vocab_size + 1, 1, padding_idx=0).cuda()
        self.r_noise.weight.requires_grad = False
        
        self.alpha = alpha
        
    def forward(self, feat_idx):
        w = self.w(feat_idx) + 0.4
        r = (self.r(feat_idx) * (1 - self.alpha)) + (self.r_noise(feat_idx) * self.alpha)
        
        r = torch.cat([r, -r], dim=-1)
        
        x = self.fc((w * r).sum(1))
        return x


def make_net(r_emb, alpha):
    return DotProdNB(vocab_size, r_emb, alpha)


# --
# Parameters

hyper_lr = 0.1
init_lr = 0.1
init_mo = -2
fix_init = True
fix_data = False
meta_iters = 250

num_iters = 50
batch_size = 64

# --
# Parameterize learning rates + momentum

n_groups = 2

r = calc_r(0, X_train, y_train) / 10
r = np.pad(r, (1, 0), mode='constant')
r = torch.FloatTensor(r).cuda().contiguous().view(-1, 1)

r_emb = nn.Embedding(vocab_size + 1, 1).cuda()
r_emb.weight.data.set_(r)

lr_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
mo_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)

# --
# Hyper-optimizer

mts = Variable(torch.FloatTensor([0.5]).view(-1, 1).cuda(), requires_grad=True)
hopt = torch.optim.Adam([mts], lr=hyper_lr)

# --
# Run

set_seeds(123)
hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    print('meta_iter=%d' % meta_iter)
    
    # Transform hyperparameters
    lrs = torch.clamp(lr_mean, 0.001, 10.0).expand((num_iters, n_groups))
    mos = torch.clamp(1 - 10 ** (mo_mean), min=0, max=1).expand((num_iters, n_groups))
    
    # Do hyperstep
    hopt.zero_grad()
    if fix_init:
        set_seeds(123)
    
    net = make_net(r_emb, mts)
    
    params = list(net.w.parameters()) + list(net.fc.parameters())
    h = HyperLayer(
        X_train_,
        y_train_,
        num_iters, 
        batch_size, 
        seed=0 if fix_data else meta_iter
    )
    h(net, lrs, mos, params=params, mts=mts, val_data=(X_test_, y_test_)) # !! This number is a meaningless hack to get gradients flowing
    hopt.step()
    
    # Log
    print('print val_acc=%f | loss_hist.tail.mean=%f | acc_hist.tail.mean=%f' % (
        h.val_acc,
        h.loss_hist[-10:].mean(),
        h.acc_hist[-10:].mean(),
    ))
    
    hist['val_acc'].append(h.val_acc)
    hist['loss_hist'].append(h.loss_hist)
    hist['acc_hist'].append(h.acc_hist)
    hist['lrs'].append(to_numpy(lrs))
    hist['mos'].append(to_numpy(mos))
    
    # _ = plt.plot(to_numpy(lrs).squeeze())
    # _ = plt.plot(to_numpy(mos).squeeze())
    # show_plot()
    print(to_numpy(mts))