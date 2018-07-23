#!/usr/bin/env python

"""
    cnn.py
"""

import os
import sys
import json
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('.')
from helpers import to_numpy, set_seeds
from hyperlayer import HyperLayer

set_seeds(123)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# IO

if not os.path.exists('.data/cnn_X_train'):
    from keras.datasets import mnist
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
    X_train = torch.FloatTensor(X_train.astype('float')).cuda()
    y_train = torch.LongTensor(y_train.astype('int')).cuda()
    
    X_val = torch.FloatTensor(X_val.astype('float')).cuda()
    y_val = torch.LongTensor(y_val.astype('int')).cuda()
    
    X_train = X_train.view(X_train.size(0), 1, 28, 28)
    X_val = X_val.view(X_val.size(0), 1, 28, 28)
    
    X_train = X_train.expand(X_train.size(0), 3, 28, 28)
    X_val = X_val.expand(X_val.size(0), 3, 28, 28)
    
    X_train, X_val = X_train / 255, X_val / 255
    
    torch.save(X_train, open('.data/cnn_X_train', 'wb'))
    torch.save(y_train, open('.data/cnn_y_train', 'wb'))
    torch.save(X_val, open('.data/cnn_X_val', 'wb'))
    torch.save(y_val, open('.data/cnn_y_val', 'wb'))
else:
    X_train = torch.load(open('.data/cnn_X_train', 'rb'))
    y_train = torch.load(open('.data/cnn_y_train', 'rb'))
    X_val = torch.load(open('.data/cnn_X_val', 'rb'))
    y_val = torch.load(open('.data/cnn_y_val', 'rb'))


# --
# Define network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1   = nn.Linear(32 * 12 ** 2, 128)
        self.fc2   = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 12 ** 2)
        # x = F.dropout(x, p=0.25, training=self.training) # !! Supported?
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training) # !! Supported?
        x = self.fc2(x)
        return x

# --
# Parameters

num_iters  = 128
batch_size = 512

hyper_lr   = 0.005
init_lr    = 0.1
init_mo    = 0.0001
fix_init   = True
fix_data   = True
meta_iters = 1000

# --
# Parameterize learning rates + momentum

szs = [sum([np.prod(p.size()) for p in Net().parameters()])]

lr_mean = torch.tensor([init_lr]).cuda().requires_grad_()
lr_res  = torch.zeros(num_iters).cuda().requires_grad_()

# mo_mean = torch.tensor([init_mo]).cuda().requires_grad_()
# mo_res  = torch.zeros(num_iters).cuda().requires_grad_()

lr_mean = lr_mean.cuda().requires_grad_()
lr_res  = lr_res.cuda().requires_grad_()

# mo_mean = mo_mean.cuda().requires_grad_()
# mo_res  = mo_res.cuda().requires_grad_()

# --
# Hyper-optimizer

hopt = torch.optim.Adam([lr_mean, lr_res], lr=hyper_lr)

# --
# Run

set_seeds(123)
hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    print('meta_iter=%d' % meta_iter, file=sys.stderr)
    
    # --
    # Transform hyperparameters
    
    lr_shape = torch.cat([
        torch.linspace(0, 1, int(num_iters / 2)),
        torch.linspace(1, 0, int(num_iters / 2)),
    ]).cuda()
    lrs = torch.clamp(lr_shape * lr_mean + lr_res, 0.001, 10.0).view(-1, 1)
    # mos = 1 - 10 ** torch.clamp(mo_mean + mo_res, -5, -0.001).view(-1, 1)
    mos = 1e-5 + torch.zeros(num_iters, 1).cuda().requires_grad_()
    
    # --
    # Hyperstep
    
    hopt.zero_grad()
    if fix_init:
        set_seeds(123)
    
    net = Net().cuda()
    
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

for mo in to_numpy(mos).T:
    _ = plt.plot(mo)

show_plot()

_ = plt.plot([h[-1] for h in hist['acc_hist']])
show_plot()



