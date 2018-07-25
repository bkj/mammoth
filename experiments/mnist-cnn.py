#!/usr/bin/env python

"""
    mnist-cnn.py
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('../mammoth')
from mammoth.utils import load_data
from mammoth.helpers import to_numpy, set_seeds
from mammoth.hyperlayer import HyperLayer
from mammoth.optim import LambdaAdam

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
set_seeds(seed)

# --
# IO

if not os.path.exists('./data/prepped_mnist'):
    transform = transforms.ToTensor()
    
    train_data = MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    X_train, y_train = zip(*train_data)
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    
    mean = X_train.mean()
    std  = X_train.std()
    
    X_train -= mean
    X_train /= std
    
    valid_data  = MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    X_valid, y_valid = zip(*valid_data)
    X_valid = torch.stack(X_valid)
    y_valid = torch.stack(y_valid)
    
    X_valid -= mean
    X_valid /= std
    
    valid_idx, test_idx = train_test_split(range(X_valid.shape[0]), train_size=0.5)
    valid_idx, test_idx = sorted(valid_idx), sorted(test_idx)
    
    X_valid, X_test = X_valid[valid_idx], X_valid[test_idx]
    y_valid, y_test = y_valid[valid_idx], y_valid[test_idx]
    
    torch.save((X_train, X_valid, X_test, y_train, y_valid, y_test), './data/prepped_mnist')


X_train, X_valid, X_test, y_train, y_valid, y_test = [x.cuda() for x in torch.load('./data/prepped_mnist')]
X_train, X_valid, X_test = X_train.float(), X_valid.float(), X_test.float()
y_train, y_valid, y_test = y_train.long(), y_valid.long(), y_test.long()

assert X_train.mean() < 1e-6
assert (X_train.std() - 1).abs() < 1e-6

# --
# Helpers

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc1 = nn.Linear(64 * 12 ** 2, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 64 * 12 ** 2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# --
# Parameters

num_iters  = 100
batch_size = 100
verbose    = True
hyper_lr   = 0.05
init_lr    = -1
init_mo    = -1
meta_iters = 20

# --
# Run

set_seeds(seed + 1)

n_groups = len(list(Net().parameters()))
hparams = {
    "lr"  : torch.FloatTensor(np.full((num_iters, n_groups), init_lr)),
    "mo"  : torch.FloatTensor(np.full((num_iters, n_groups), init_mo)),
}

for k,v in hparams.items():
    hparams[k] = v.float().cuda().requires_grad_()

hopt = LambdaAdam(
    params=list(hparams.values()),
    lr=hyper_lr,
    lam=1,
)

hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    
    net = Net().cuda()
    
    _ = hopt.zero_grad()
    hlayer = HyperLayer(
        net=net, 
        hparams={
            "lrs"  : 10 ** hparams['lr'],
            "mos"  : 1 - (10 ** hparams['mo']),
            "meta" : None,
        },
        params=net.parameters(),
        num_iters=num_iters, 
        batch_size=batch_size,
        seed=(seed * meta_iter),
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

for lr in to_numpy(10 ** hparams['lr']).T:
    _ = plt.plot(lr)

show_plot()

