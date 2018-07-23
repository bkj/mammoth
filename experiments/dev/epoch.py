#!/usr/bin/env python

"""
    main-epoch.py
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

from hypergrad.data import load_data

from rsub import *
from matplotlib import pyplot as plt

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
# Params

hyper_lr = 0.05
init_lr = 0.10
init_mo = 0.50
fix_data = False
meta_iters = 50

_ = torch.manual_seed(123)
_ = torch.cuda.manual_seed(123)

n_groups = len(list(make_net().parameters()))

# --
# Optimizing linear decay w/ single LR and fixed MO

lr_max = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
mo = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)
c = Variable(1 - torch.arange(0, num_iters).view(-1, 1) / num_iters).cuda()

hopt = torch.optim.Adam([lr_max, mo], lr=hyper_lr)

hist = defaultdict(list)

for meta_iter in range(0, meta_iters):
    print 'meta_iter=%d' % meta_iter
    print 'lr=', to_numpy(lr_max).squeeze()
    print 'mo=', to_numpy(mo).squeeze()
    
    # Transform hyperparameters
    lrs = lr_max * c
    mos = torch.clamp(mo, 0.001, 0.999).repeat(num_iters, 1)
    
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
    hist['lrs'].append(to_numpy(lrs))
    hist['mos'].append(to_numpy(mos))


for l in to_numpy(lrs).T:
    _ = plt.plot(l, alpha=0.25)

show_plot()

_ = plt.plot(np.hstack(hist['val_acc']))
_ = plt.ylim(0.8, 1.0)
show_plot()

# Training further w/ learned schedule
net = make_net().cuda()
for meta_iter in range(10):
    lrs = lr_max * c * (1 - float(meta_iter) / 10)
    mos = torch.clamp(mo, 0.001, 0.999).repeat(num_iters, 1)
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
    _ = h(net, lrs, mos, val_data=(X_val, y_val))
    print h.val_acc # 0.9652



lr_max # 0.6906

net = make_net().cuda()
lrs = init_lr * c
mos = torch.clamp(mo_init, 0.001, 0.999).repeat(num_iters, 1)
for meta_iter in range(10):
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
    _ = h(net, lrs, mos, val_data=(X_val, y_val), szs=szs)
    print h.val_acc # 0.9603



# --
# Optimizing

szs = [sum([np.prod(p.size()) for p in make_net().parameters()])]

lr_max = Variable(torch.FloatTensor(np.full((1, 1), init_lr)).cuda(), requires_grad=True)
mo = Variable(torch.FloatTensor(np.full((1, 1), init_mo)).cuda(), requires_grad=True)
c = Variable(1 - torch.arange(0, num_iters).view(-1, 1) / num_iters).cuda()

hopt = torch.optim.Adam([lr_max, mo], lr=hyper_lr)

hist = defaultdict(list)

for meta_iter in range(0, meta_iters):
    print 'meta_iter=%d' % meta_iter
    print 'lr=%f' % to_numpy(lr_max).squeeze()
    print 'mo=%f' % to_numpy(mo).squeeze()
    
    # Transform hyperparameters
    lrs = lr_max * c
    mos = torch.clamp(mo, 0.001, 0.999).repeat(num_iters, 1)
    
    # Do hyperstep
    hopt.zero_grad()
    net = make_net().cuda()
    h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
    dummy_loss = h(net, lrs, mos, val_data=(X_val, y_val), szs=szs)
    
    loss = dummy_loss
    loss.backward()
    hopt.step()
    
    print 'print val_acc=%f | loss_hist.tail.mean=%f | acc_hist.tail.mean=%f' % (
        h.val_acc,
        h.loss_hist[-10:].mean(),
        h.acc_hist[-10:].mean(),
    )
    
    hist['val_acc'].append(h.val_acc)
    hist['lrs'].append(to_numpy(lrs))
    hist['mos'].append(to_numpy(mos))

_ = plt.plot(np.hstack(hist['lrs'])[0])
_ = plt.plot(np.hstack(hist['mos'])[0])
_ = plt.plot(np.hstack(hist['val_acc']))
show_plot()

_ = plt.plot(to_numpy(lrs))
show_plot()


