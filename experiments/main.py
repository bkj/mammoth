#!/usr/bin/env python

"""
    main.py
"""

import sys
sys.path.append('/home/bjohnson/software/hypergrad')
from hypergrad.data import load_data
# ^^ Should remove this dependency at some point, but whatever for now

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.autograd import Variable

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('..')
from helpers import to_numpy
from hyperlayer import HyperLayer

np.random.seed(123)
_ = torch.manual_seed(456)
_ = torch.cuda.manual_seed(789)

# --
# IO

batch_size = 200
num_iters = 100

X_train, y_train, X_val, y_val, _ = load_data(normalize=True)

X_train = Variable(torch.FloatTensor(X_train)).cuda()
y_train = Variable(torch.LongTensor(y_train.argmax(axis=1))).cuda()

X_val = Variable(torch.FloatTensor(X_val)).cuda()
y_val = Variable(torch.LongTensor(y_val.argmax(axis=1))).cuda()

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

hyper_lr = 0.01
init_lr = 0.10
init_lr = 0.30
init_mo = 0.50
fix_data = False
meta_iters = 200

_ = torch.manual_seed(123)
_ = torch.cuda.manual_seed(123)

n_groups = len(list(make_net().parameters()))

lr_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
lr_res  = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)

mo_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)
mo_res  = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)

hopt = torch.optim.Adam([lr_mean, lr_res, mo_mean, mo_res], lr=hyper_lr)

hist = defaultdict(list)

for meta_iter in range(0, meta_iters):
    print 'meta_iter=%d' % meta_iter
    
    # Transform hyperparameters
    lrs = torch.clamp(lr_mean + lr_res, 0.001, 10.0)
    mos = torch.clamp(mo_mean + mo_res, 0.001, 0.999)
    
    # Do hyperstep
    hopt.zero_grad()
    net = make_net().cuda()
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
    dummy_loss = h(net, lrs, mos, val_data=(X_val, y_val))
    
    loss = dummy_loss
    loss.backward()
    hopt.step()
    
    print 'print val_acc=%f | loss_hist.tail.mean=%f | acc_hist.tail.mean=%f' % (
        h.val_acc,
        h.loss_hist[-10:].mean(),
        h.acc_hist[-10:].mean(),
    )
    
    hist['val_acc'].append(h.val_acc)
    hist['loss_hist'].append(h.loss_hist)
    hist['acc_hist'].append(h.acc_hist)
    hist['lrs'].append(to_numpy(lrs))
    hist['mos'].append(to_numpy(mos))


_ = plt.plot(hist['val_acc'])
show_plot()

for l in hist['lrs'][-1].T[::2]:
    _ = plt.plot(l)

show_plot()

_ = plt.plot(h.acc_hist)
show_plot()
