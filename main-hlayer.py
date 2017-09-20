#!/usr/bin/env python

"""
    hlayer.py
"""

import sys
sys.path.append('/home/bjohnson/software/hypergrad')

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.autograd import Variable

from hypergrad.data import load_data_dicts

from rsub import *
from matplotlib import pyplot as plt

from helpers import to_numpy
from hlayer import HyperLayer

np.random.seed(123)
_ = torch.manual_seed(456)
_ = torch.cuda.manual_seed(789)

# --
# IO

batch_size  = 200
num_iters   = 100
N_classes   = 10
N_train     = 10000
N_valid     = 10000
N_tests     = 10000

train_data, valid_data, _ = load_data_dicts(N_train, N_valid, N_tests)

X_train = Variable(torch.FloatTensor(train_data['X'])).cuda()
y_train = Variable(torch.LongTensor(train_data['T'].argmax(axis=1))).cuda()

X_val = Variable(torch.FloatTensor(valid_data['X'])).cuda()
y_val = Variable(torch.FloatTensor(valid_data['T'])).cuda()

# --
# Helpers

def make_net(weight_scale=np.exp(-3), layers=[50, 50, 50]):
    
    net = nn.Sequential(
        nn.Linear(784, layers[0]),
        nn.Tanh(),
        nn.Linear(layers[0], layers[1]),
        nn.Tanh(),
        nn.Linear(layers[1], layers[2]),
        nn.Tanh(),
        nn.Linear(layers[2], 10),
    )
    
    for child in net.children():
        if isinstance(child, nn.Linear):
            _ = child.weight.data.normal_(0, weight_scale)
    
    return net

# --
# Run

lr_mean = Variable(torch.FloatTensor(np.full((1, 8), 0.01)).cuda(), requires_grad=True)
lr_res  = Variable(torch.FloatTensor(np.full((num_iters, 8), 0.0)).cuda(), requires_grad=True)

mo_mean  = Variable(torch.FloatTensor(np.full((1, 8), 0.5)).cuda(), requires_grad=True)
mo_res  = Variable(torch.FloatTensor(np.full((num_iters, 8), 0.0)).cuda(), requires_grad=True)

hopt = torch.optim.Adam([lr_mean, lr_res, mo_mean, mo_res], lr=0.01)

hist = defaultdict(list)

for meta_iter in range(50):
    
    # Transform hyperparameters
    lrs_ = torch.clamp(lr_mean + lr_res, 0.001, 10.0)
    mos_ = torch.clamp(mo_mean + mo_res, 0.001, 0.999)
    
    # Do hyperstep
    hopt.zero_grad()
    net = make_net().cuda()
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=meta_iter)
    dummy_loss = h(net, lrs_, mos_, val_data=(X_val, y_val))
    print 'val_acc=%f' % h.val_acc
    
    loss = dummy_loss
    loss.backward()
    hopt.step()
    
    hist['val_acc'].append(h.val_acc)


