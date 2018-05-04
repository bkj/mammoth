#!/usr/bin/env python

"""
    main.py
    
    Example of running per-step learning rate and momentum tuning
    on a toy MNIST network
"""

from __future__ import print_function

import sys
import json
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('/home/bjohnson/software/hypergrad')
from hypergrad.data import load_data
# ^^ Should remove this dependency at some point, but whatever for now

sys.path.append('.')
from helpers import to_numpy, set_seeds
from hyperlayer import HyperLayer

set_seeds(123)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# IO

X_train, y_train, X_val, y_val, _ = load_data(normalize=True)

X_train = torch.FloatTensor(X_train).cuda()
y_train = torch.LongTensor(y_train).argmax(dim=-1).cuda()

X_val = torch.FloatTensor(X_val).cuda()
y_val = torch.LongTensor(y_val).argmax(dim=-1).cuda()


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
# Parameters

num_iters  = 200
batch_size = 200

hyper_lr   = 0.01
init_lr    = 0.30
init_mo    = 0.50
fix_init   = True
fix_data   = True
meta_iters = 250

# --
# Parameterize learning rates + momentum

n_groups = len(list(make_net().parameters()))

szs = [sum([np.prod(p.size()) for p in make_net().parameters()])]
lr_init = torch.tensor(np.full((1, 1), init_lr)).cuda().requires_grad_()
lr_res  = torch.tensor(np.full((1, num_iters), 0.0)).cuda().requires_grad_()
mo_init = torch.tensor(np.full((1, 1), init_mo)).cuda().requires_grad_()

# --
# Hyper-optimizer

hopt = torch.optim.Adam([lr_init, mo_init], lr=hyper_lr)

# --
# Run

set_seeds(123)
hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    print('meta_iter=%d' % meta_iter, file=sys.stderr)
    
    # --
    # Transform hyperparameters
    
    # lin = Variable(torch.linspace(1, 0, num_iters).expand(1, num_iters).t()).cuda()
    lrs = torch.clamp(lr_init, 0.001, 10.0) + lr_res.expand(1, num_iters).t()
    mos = torch.clamp(mo_init, 0.001, 0.999).expand(num_iters, 1)
    
    # --
    # Hyperstep
    
    hopt.zero_grad()
    if fix_init:
        set_seeds(123)
    
    net = make_net().cuda()
    
    # params = list(net.parameters())
    
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
    h(net, lrs, mos, val_data=(X_val, y_val), szs=szs)
    hopt.step()
    
    # --
    # Logging
    
    hist['val_acc'].append(h.val_acc)
    hist['loss_hist'].append(h.loss_hist)
    hist['acc_hist'].append(h.acc_hist)
    hist['lrs'].append(to_numpy(lrs))
    hist['mos'].append(to_numpy(mos))
    
    print(json.dumps({
        "val_acc"        : float(h.val_acc),
        "tail_loss_mean" : float(h.loss_hist[-10:].mean()),
        "tail_acc_mean"  : float(h.acc_hist[-10:].mean()),
    }))
    
    for lr in to_numpy(lrs).T:
        _ = plt.plot(lr)
        
    show_plot()

_ = plt.plot([h[-1] for h in hist['acc_hist']])
show_plot()



