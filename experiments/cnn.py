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
num_iters   = 468 * 4

if not os.path.exists('.cnn_X_train'):
    from keras.datasets import mnist
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
    X_train = Variable(torch.FloatTensor(X_train.astype('float'))).cuda()
    y_train = Variable(torch.LongTensor(y_train.astype('int'))).cuda()
    
    X_val = Variable(torch.FloatTensor(X_val.astype('float'))).cuda()
    y_val = Variable(torch.LongTensor(y_val.astype('int'))).cuda()
    
    X_train = X_train.view(X_train.size(0), 1, 28, 28)
    X_val = X_val.view(X_val.size(0), 1, 28, 28)
    
    X_train = X_train.expand(X_train.size(0), 3, 28, 28)
    X_val = X_val.expand(X_val.size(0), 3, 28, 28)
    
    X_train, X_val = X_train / 255, X_val / 255
    
    torch.save(X_train, open('.cnn_X_train', 'w'))
    torch.save(y_train, open('.cnn_y_train', 'w'))
    torch.save(X_val, open('.cnn_X_val', 'w'))
    torch.save(y_val, open('.cnn_y_val', 'w'))
else:
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

# --
# Params

hyper_lr = 0.01
init_lr = 0.20
init_mo = 0.50
fix_data = False
meta_iters = 50

_ = torch.manual_seed(123)
_ = torch.cuda.manual_seed(123)

n_groups = len(list(Net().parameters()))

# --
# Hypertraining

n_groups = len(list(Net().cuda().parameters()))

lr_max = Variable(torch.FloatTensor(np.full((1, n_groups), init_lr)).cuda(), requires_grad=True)
mo = Variable(torch.FloatTensor(np.full((1, n_groups), init_mo)).cuda(), requires_grad=True)
c = Variable(1 - torch.arange(0, num_iters).view(-1, 1) / num_iters).cuda()

hopt = torch.optim.Adam([lr_max, mo], lr=hyper_lr)

hist = defaultdict(list)

for meta_iter in range(0, meta_iters):
    try:
        print 'meta_iter=%d' % meta_iter
        print 'lr=', to_numpy(lr_max).squeeze()
        print 'mo=', to_numpy(mo).squeeze()
        
        # Transform hyperparameters
        lrs = torch.clamp(lr_max * c, 0, 999)
        mos = torch.clamp(mo, 0.001, 0.999).repeat(num_iters, 1)
        
        # Do hyperstep
        hopt.zero_grad()
        net = Net().cuda()
        h = HyperLayer(X_train, y_train, num_iters, batch_size, seed=meta_iter)
        dummy_loss = h(net, lrs, mos, val_data=(X_val, y_val))
        
        # reg_loss = reg_strength * F.relu(lr_res[1:] - lr_res[:-1] - 0.01).sum()
        
        loss = dummy_loss# + reg_loss
        loss.backward()
        hopt.step()
        
        print 'print val_acc=%f | loss_hist.tail.mean=%f | acc_hist.tail.mean=%f | reg_loss=%f' % (
            h.val_acc,
            h.loss_hist[-10:].mean(),
            h.acc_hist[-10:].mean(),
            # to_numpy(reg_loss)[0],
            0
        )
        
        hist['val_acc'].append(h.val_acc)
        hist['lrs'].append(to_numpy(lrs))
        hist['mos'].append(to_numpy(mos))
    except KeyboardInterrupt:
        raise
    except:
        print "nonexact backward pass at meta_iter=%d -- skipping" % meta_iter



for l in hist['lrs'][-1].T[::2]:
    _ = plt.plot(l)

show_plot()


_ = plt.plot(np.hstack(hist['val_acc']))
show_plot()
