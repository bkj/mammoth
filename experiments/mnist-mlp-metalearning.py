#!/usr/bin/env python

"""
    main.py
    
    Example of running per-step learning rate and momentum tuning
    on a toy MNIST network
"""

import sys
import json
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('../mammoth')
from mammoth.utils import load_data
from mammoth.helpers import to_numpy, set_seeds
from mammoth.hyperlayer import HyperLayer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# IO

set_seeds(123)

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

X_train = torch.FloatTensor(X_train).cuda()
X_valid = torch.FloatTensor(X_valid).cuda()
X_test  = torch.FloatTensor(X_test).cuda()

y_train = torch.LongTensor(y_train).cuda()
y_valid = torch.LongTensor(y_valid).cuda()
y_test  = torch.LongTensor(y_test).cuda()

# --
# Helpers

class Net(nn.Module):
    def __init__(self, layers=[50, 50, 50]):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(784, layers[0]),
            nn.Tanh(),
            nn.Linear(layers[0], layers[1]),
            nn.Tanh(),
            nn.Linear(layers[1], layers[2]),
            nn.Tanh(),
            nn.Linear(layers[2], 10),
        )
    
    def forward(self, x):
        return self.layers(x)


# --
# Parameters

num_iters  = 200
batch_size = 200

hyper_lr   = 0.01
init_lr    = 0.30
init_mo    = 0.50
fix_init   = False
fix_data   = False
meta_iters = 20

# --
# Parameterize learning rates + momentum

n_groups = len(list(Net().parameters()))

lr_mean = torch.tensor(np.full((1, n_groups), init_lr)).cuda().requires_grad_()
lr_res  = torch.tensor(np.full((num_iters, n_groups), 0.0)).cuda().requires_grad_()

mo_mean = torch.tensor(np.full((1, n_groups), init_mo)).cuda().requires_grad_()
mo_res  = torch.tensor(np.full((num_iters, n_groups), 0.0)).cuda().requires_grad_()

# --
# Hyper-optimizer

# --
# Run

set_seeds(123)

net = Net().cuda()

hparams = list(net.parameters()) + [lr_mean, lr_res, mo_mean, mo_res]
hopt = torch.optim.Adam(
    params=hparams, 
    lr=hyper_lr
)

hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    
    # --
    # Transform hyperparameters
    
    lrs = torch.clamp(lr_mean + lr_res, 0.001, 10.0)
    mos = torch.clamp(mo_mean + mo_res, 0.001, 0.999)
    
    # --
    # Hyperstep
    
    _ = hopt.zero_grad()
    hlayer = HyperLayer(
        net=net, 
        num_iters=num_iters, 
        batch_size=batch_size,
        seed=0 if fix_data else meta_iter
    )
    train_hist, val_acc, test_acc = hlayer.run(
        X_train=X_train,
        y_train=y_train, 
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        
        lrs=lrs,
        mos=mos,
        mts=None,
        untrain=True,
        update_weights=True,
    )
    _ = hopt.step()
    
    # --
    # Logging
    
    hist['train_hist'].append(train_hist)
    hist['val_acc'].append(val_acc)
    hist['test_acc'].append(test_acc)
    
    print(json.dumps({
        "meta_iter" : meta_iter,
        "train_acc" : float(np.mean([t['acc'] for t in train_hist[-10:]])),
        "val_acc"   : val_acc,
        "test_acc"  : test_acc,
    }))
    sys.stdout.flush()


# --
# Plot results

_ = plt.plot(hist['val_acc'], label='val_acc')
_ = plt.plot(hist['test_acc'], label='test_acc')
_ = plt.legend()
show_plot()

# for lr in to_numpy(lrs).T:
#     _ = plt.plot(lr)

# show_plot()

