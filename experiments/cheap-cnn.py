#!/usr/bin/env python

"""
    cnn-hlayer.py
"""

import os
import sys
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

from helpers import to_numpy
from hyperlayer import HyperLayer

np.random.seed(123)
_ = torch.manual_seed(456)
_ = torch.cuda.manual_seed(789)

if torch.__version__ != '0.2.0+9b8f5eb_dev':
    os._exit(1)
else:
    # !! Ordinarily forces use of best algorithm, but hacked to use default (determnistic) ops
    torch.backends.cudnn.benchmark = True

# --
# IO

batch_size  = 128
num_iters   = 60000 / 128

# from keras.datasets import mnist
# (X_train, y_train), (X_val, y_val) = mnist.load_data()

# X_train = Variable(torch.FloatTensor(X_train.astype('float'))).cuda()
# y_train = Variable(torch.LongTensor(y_train.astype('int'))).cuda()

# X_val = Variable(torch.FloatTensor(X_val.astype('float'))).cuda()
# y_val = Variable(torch.LongTensor(y_val.astype('int'))).cuda()

# X_train = X_train.view(X_train.size(0), 1, 28, 28)
# X_val = X_val.view(X_val.size(0), 1, 28, 28)

# X_train = X_train.expand(X_train.size(0), 3, 28, 28)
# X_val = X_val.expand(X_val.size(0), 3, 28, 28)

# X_train, X_val = X_train / 255, X_val / 255

# torch.save(X_train, open('.cnn_X_train', 'w'))
# torch.save(y_train, open('.cnn_y_train', 'w'))
# torch.save(X_val, open('.cnn_X_val', 'w'))
# torch.save(y_val, open('.cnn_y_val', 'w'))

X_train = torch.load(open('.cnn_X_train'))
y_train = torch.load(open('.cnn_y_train'))
X_val = torch.load(open('.cnn_X_val'))
y_val = torch.load(open('.cnn_y_val'))

# --
# Define network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1   = nn.Linear(64 * 12 ** 2, 128)
        self.fc2   = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 ** 2)
        # x = F.dropout(x, p=0.25, training=self.training) # !! Supported?
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training) # !! Supported?
        x = self.fc2(x)
        return x

# Train a model to convergence, with a good optimizer

# --
# Params

hyper_lr = 0.01
init_lr = 0.20
init_mo = 0.50
fix_data = False
meta_iters = 50

_ = torch.manual_seed(123)
_ = torch.cuda.manual_seed(123)

# --
# Hypertraining

n_groups = len(list(Net().cuda().parameters()))

lr_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
lr_res = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)

mo_mean = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)
mo_res = Variable(torch.FloatTensor(np.full((num_iters, n_groups), 0.0)).cuda(), requires_grad=True)

hopt = torch.optim.Adam([lr_mean, lr_res, mo_mean, mo_res], lr=hyper_lr)

hist = defaultdict(list)

for meta_iter in range(0, 1000):
    # try:
        # Set seed for reproducibility
        seed = np.random.choice(10000)
        _ = torch.manual_seed(seed)
        _ = torch.cuda.manual_seed(seed)
        
        print 'meta_iter=%d' % meta_iter
        
        # Transform hyperparameters
        # !! Having problems w/ dying units here
        lrs = torch.clamp(lr_mean + lr_res, -10, 10.0)
        mos = torch.clamp(mo_mean + mo_res, -10, 0.999)
        
        # Hyperstep
        hopt.zero_grad()
        net = Net().cuda()
        h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=0 if fix_data else meta_iter)
        dummy_loss = h(net, lrs, mos, val_data=(X_val, y_val), cheap=True)
        
        loss = dummy_loss
        loss.backward()
        hopt.step()
        
        print 'print val_acc=%f | loss_hist.tail.mean=%f | acc_hist.tail.mean=%f' % (
            h.val_acc,
            h.loss_hist[-10:].mean(),
            h.acc_hist[-10:].mean()
        )
        
        hist['val_acc'].append(h.val_acc)
        hist['lrs'].append(to_numpy(lrs))
        hist['mos'].append(to_numpy(mos))
    # except KeyboardInterrupt:
    #     raise
    # except:
    #     print "assertion error @ %d" % meta_iter
    #     pass

# import cPickle
# cPickle.dump(hist, open('./.cheap_hist', 'w'))

for l in to_numpy(mos).T[::2]:
    _ = plt.plot(l)

show_plot()

for l in hist['lrs'][30].T[::2]:
    _ = plt.plot(l)

show_plot()

np.percentile(np.hstack(hist['val_acc']), 90)

_ = plt.plot(np.hstack(hist['val_acc']))
show_plot()
