#!/usr/bin/env python

"""
    meta_test.py
    
    Show that `mammoth` can optimizer "hyperparameters" via gradient descent
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
# Fake some data

set_seeds(123)

N = 10000
X_train_ = Variable(torch.randn((N, 2)).cuda())
y_train_ = Variable(torch.ones(N).cuda())

X_test_ = X_train_.clone()
y_test_ = y_train_.clone()

# --
# Define model

class MTSTest(nn.Module):
    def __init__(self, bias):
        super(MTSTest, self).__init__()
        self.bias = bias
        self.fc = nn.Linear(2, 1, bias=False)
    
    def forward(self, x):
        return self.fc(x) + self.bias

def make_net(mts):
    return MTSTest(bias=mts[0]).cuda()

# --
# Parameters

hyper_lr   = 0.05
init_lr    = 0.5
init_mo    = -1
fix_data   = False
meta_iters = 250

num_iters  = 100
batch_size = 50

# --
# Parameterize learning rates + momentum

n_groups = 1

mts = Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True)
lr_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
lr_res  = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)
mo_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)
mo_res  = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)

# --
# Hyper-optimizer

hopt = torch.optim.Adam([lr_mean, lr_res, mo_mean, mo_res, mts], lr=hyper_lr)

# --
# Run

set_seeds(123)
hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    print('meta_iter=%d' % meta_iter)
    
    lrs = torch.clamp(lr_mean + lr_res, 0.001, 10.0)
    mos = torch.clamp(1 - 10 ** (mo_mean + mo_res), min=0.001, max=1)
    
    # Do hyperstep
    hopt.zero_grad()
    # set_seeds(123)
    net = make_net(mts)
    
    params = list(net.parameters())
    h = HyperLayer(
        X_train_,
        y_train_,
        num_iters, 
        batch_size, 
        seed=0 if fix_data else meta_iter,
        loss_function=F.mse_loss
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
    
    for h in hist['loss_hist']:
        _ = plt.plot(h)
    
    show_plot()
    
    _ = plt.plot(to_numpy(lrs).squeeze())
    show_plot()
    
    print(mts.data[0])

