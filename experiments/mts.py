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

# >>
from sklearn.model_selection import train_test_split
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5)
# <<

X_train = torch.FloatTensor(X_train).cuda()
y_train = torch.LongTensor(y_train).argmax(dim=-1).cuda()

X_val = torch.FloatTensor(X_val).cuda()
y_val = torch.LongTensor(y_val).argmax(dim=-1).cuda()

# >>
X_test = torch.FloatTensor(X_test).cuda()
y_test = torch.LongTensor(y_test).argmax(dim=-1).cuda()
# <<

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

num_iters  = 20
batch_size = 512

hyper_lr   = 0.001
init_lr    = 0.2
init_mo    = 0.9
fix_init   = False
fix_data   = False
reset_net  = False
meta_iters = 1000

# --
# Parameterize learning rates + momentum

net = make_net().cuda()
szs = [sum([np.prod(p.size()) for p in net.parameters()])]

lr_mean = torch.tensor([init_lr]).cuda().requires_grad_()
lr_res  = torch.zeros(num_iters).cuda().requires_grad_()

mo_mean = torch.tensor([init_mo]).cuda().requires_grad_()
mo_res  = torch.zeros(num_iters).cuda().requires_grad_()

lr_mean = lr_mean.cuda().requires_grad_()
lr_res  = lr_res.cuda().requires_grad_()

mo_mean = mo_mean.cuda().requires_grad_()
mo_res  = mo_res.cuda().requires_grad_()

# --
# Hyper-optimizer

params = list(net.parameters())
params += [lr_mean, lr_res, mo_mean, mo_res]
hopt = torch.optim.Adam(params, lr=hyper_lr)

# --
# Run

set_seeds(123)
hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    try:
        # --
        # Transform hyperparameters
        
        lr_shape = torch.cat([
            # torch.linspace(0, 1, int(num_iters / 2)),
            torch.linspace(1, 1, int(num_iters)),
        ]).cuda()
        lrs = torch.clamp(lr_shape * lr_mean + lr_res, 0.001, 10.0).view(-1, 1)
        mos = 1e-5 * torch.clamp(mo_mean + mo_res, 0, 10).view(-1, 1)
        
        # --
        # Hyperstep
        
        hopt.zero_grad()
        if fix_init:
            set_seeds(123)
        
        # if reset_net:
        #     net = make_net().cuda()
        
        h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter, verbose=False)
        h(net, lrs, mos, val_data=(X_val, y_val), test_data=(X_test, y_test), szs=szs, untrain=True)
        hopt.step()
        
        # --
        # Logging
        
        hist['val_acc'].append(h.val_acc)
        hist['loss_hist'].append(h.loss_hist)
        hist['acc_hist'].append(h.acc_hist)
        hist['lrs'].append(to_numpy(lrs))
        hist['mos'].append(to_numpy(mos))
        
        print(json.dumps({
            "meta_iter"      : int(meta_iter),
            "val_acc"        : float(h.val_acc),
            "test_acc"       : float(h.test_acc),
            "tail_loss_mean" : float(h.loss_hist[-10:].mean()),
            "tail_acc_mean"  : float(h.acc_hist[-10:].mean()),
        }))
    except KeyboardInterrupt:
        raise
    except:
        print('err')


# --

for lr in to_numpy(lrs).T:
    _ = plt.plot(lr)
    
for mo in to_numpy(mos).T:
    _ = plt.plot(mo)
    
show_plot()

_ = plt.plot([h[-1] for h in hist['acc_hist']])
show_plot()



