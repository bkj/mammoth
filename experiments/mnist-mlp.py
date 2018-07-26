#!/usr/bin/env python

"""
    
"""

import sys
import json
import numpy as np
from time import time
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
from mammoth.optim import LambdaAdam

# --
# IO

seed = 345
set_seeds(seed)

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

X_train = torch.FloatTensor(X_train).cuda()
X_valid = torch.FloatTensor(X_valid).cuda()
X_test  = torch.FloatTensor(X_test).cuda()

y_train = torch.LongTensor(y_train).cuda()
y_valid = torch.LongTensor(y_valid).cuda()
y_test  = torch.LongTensor(y_test).cuda()

assert X_train.mean() < 1e-3
assert (1 - X_train.std()).abs() < 1e-3

# --
# Helpers

class Net(nn.Module):
    def __init__(self, layers=[256, 64, 32]):
        super().__init__()
        
        self.layers1 = nn.Sequential(
            nn.Linear(784, layers[0]),
            nn.Tanh(),
            nn.Linear(layers[0], layers[1]),
            nn.Tanh(),
            nn.Linear(layers[1], layers[2]),
            nn.Tanh(),
            nn.Linear(layers[2], 10),
        )
    
    def forward(self, x):
        return self.layers1(x)


# --
# Parameters

num_iters  = 100
batch_size = 100
verbose    = False
hyper_lr   = 0.01
init_lr    = 0.1
init_mo    = 0.5
meta_iters = 20

# --
# Run

set_seeds(seed)

n_groups = len(list(Net().parameters()))

hparams = {
    "lr" : torch.FloatTensor(np.full((num_iters, n_groups), init_lr)),
    "mo" : torch.FloatTensor(np.full((num_iters, n_groups), init_mo)),
}

for k,v in hparams.items():
    hparams[k] = v.cuda().requires_grad_()

hopt = LambdaAdam(
    params=hparams.values(),
    lr=hyper_lr,
    lam=1,
)

hist = defaultdict(list)
t = time()
for meta_iter in range(0, meta_iters):
    
    net = Net().cuda()
    
    _ = hopt.zero_grad()
    hlayer = HyperLayer(
        net=net, 
        hparams={
            "lrs"  : hparams['lr'].clamp(min=0),
            "mos"  : hparams['mo'].clamp(min=0, max=1),
            "meta" : None,
        },
        params=net.parameters(),
        num_iters=num_iters, 
        batch_size=batch_size,
        seed=seed + meta_iter,
        verbose=verbose,
    )
    train_hist, val_acc, test_acc = hlayer.run(
        X_train=X_train,
        y_train=y_train, 
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        learn_lrs=True,
        learn_mos=True,
        learn_meta=False,
        forward_only=False,
    )
    _ = hopt.step()
    
    # --
    # Logging
    
    hist['train_hist'].append(train_hist)
    hist['val_acc'].append(val_acc)
    hist['test_acc'].append(test_acc)
    hist['hparams'].append({k:to_numpy(v.clone()) for k,v in hparams.items()})
    
    print(json.dumps({
        "meta_iter" : meta_iter,
        "train_acc" : float(np.mean([t['acc'] for t in train_hist[-10:]])),
        "val_acc"   : val_acc,
        "test_acc"  : test_acc,
        "time"      : time() - t,
    }))
    sys.stdout.flush()

# --
# Plot results

_ = plt.plot([float(h['lr'][0][0]) for h in hist['hparams']], label='lr')
_ = plt.plot([float(h['mo'][0][0]) for h in hist['hparams']], label='mo')
_ = plt.legend()
show_plot()

_ = plt.plot(hist['val_acc'], label='val_acc')
_ = plt.plot(hist['test_acc'], label='test_acc')
_ = plt.legend()
show_plot()

param_names = [k[0] for k in net.named_parameters()]
for i, lr in enumerate(to_numpy(lrs).T):
    if 'weight' in param_names[i]:
        _ = plt.plot(lr, label=param_names[i])

_ = plt.legend(fontsize=8)
show_plot()

