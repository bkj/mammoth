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
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('../mammoth')
from mammoth.utils import load_data
from mammoth.helpers import to_numpy, set_seeds
from mammoth.hyperlayer import HyperLayer
from mammoth.optim import LambdaAdam

sys.path.append('experiments/frog/fashion_mnist')
from model import Architecture, Network

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

set_seeds(123)

# --
# IO

if not os.path.exists('.fashion_mnist'):
    
    dataset_stats = {
        'fashion_mnist' : {
            "mean" : (0.28604060411453247,),
            "std"  : (0.3530242443084717,),
        }
    }
    
    def NormalizeDataset(dataset):
        assert dataset in set(dataset_stats.keys()), 'unknown dataset %s' % dataset
        return transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std'])
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        NormalizeDataset(dataset='fashion_mnist'),
    ])
    
    train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    X_train, y_train = zip(*train_data)
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    
    valid_data  = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    X_valid, y_valid = zip(*valid_data)
    X_valid = torch.stack(X_valid)
    y_valid = torch.stack(y_valid)
    
    valid_idx, test_idx = train_test_split(range(X_valid.shape[0]), train_size=0.5)
    
    X_valid, X_test = X_valid[valid_idx], X_valid[test_idx]
    y_valid, y_test = y_valid[valid_idx], y_valid[test_idx]
    
    torch.save((X_train, X_valid, X_test, y_train, y_valid, y_test), '.fashion_mnist')


X_train, X_valid, X_test, y_train, y_valid, y_test = [x.cuda() for x in torch.load('.fashion_mnist')]
X_train, X_valid, X_test = X_train.float(), X_valid.float(), X_test.float()
y_train, y_valid, y_test = y_train.long(), y_valid.long(), y_test.long()

X_valid, y_valid = X_valid, y_valid
X_test, y_test   = X_test, y_test

# --
# Helpers

def make_net(normal):
    net = Network(
        in_channels=1,
        num_classes=10,
        op_channels=64,
        num_layers=1,
        num_nodes=4,
    )
    
    arch = Architecture(num_nodes=4, normal=normal.view(14, 4)).cuda()
    net.init_search(arch=arch, unrolled=False)
    
    return net


# --
# Parameters

num_iters  = 300
batch_size = 100
verbose    = True

seed       = 345
hyper_lr   = 0.05
init_lr    = 0.1
init_mo    = 0.9
fix_init   = False
fix_data   = False
meta_iters = 20


# --
# Parameterize learning rates + momentum

# arch = Architecture(num_nodes=4).cuda()
# normal = arch._arch_params[0].view(-1)
normal = torch.Tensor(np.random.normal(0, 1e-3, (14 * 4,)))

n_groups = len(list(make_net(normal=normal).parameters()))

# # Fixed, same across layers
# lr_mean  = torch.FloatTensor(np.full((1, 1), init_lr))
# mo_mean  = torch.FloatTensor(np.full((1, 1), init_mo))

# # Fixed per layer
# lr_mean  = torch.FloatTensor(np.full((1, n_groups), init_lr))
# mo_mean  = torch.FloatTensor(np.full((1, n_groups), init_mo))

# Totally learned
lr_mean  = torch.FloatTensor(np.full((1, n_groups), init_lr))
mo_mean  = torch.FloatTensor(np.full((1, n_groups), init_mo))
lr_res   = torch.FloatTensor(np.full((num_iters, n_groups), 0.0))
mo_res   = torch.FloatTensor(np.full((num_iters, n_groups), 0.0))

# --
# Run

set_seeds(seed)

hparams = {
    "lr_mean" : lr_mean,
    "lr_res"  : lr_res,
    "mo_mean" : mo_mean,
    "mo_res"  : mo_res,
    "normal"  : normal,
}

for k,v in hparams.items():
    hparams[k] = v.float().cuda().requires_grad_()

# hopt = LambdaAdam(
#     params=list(hparams.values()),# + list(net.parameters()), 
#     lr=hyper_lr,
#     # lam=1e-3,
#     lam=1,
# )

net = make_net(normal=hparams['normal']).cuda()
print(net)

hist = defaultdict(list)
for meta_iter in range(0, meta_iters):
    
    # --
    # Transform hyperparameters
    
    # Fixed, same across layers
    # lrs = torch.clamp(hparams['lr_mean'].repeat(num_iters, n_groups), 0.001, 10.0)
    # mos = torch.clamp(hparams['mo_mean'].repeat(num_iters, n_groups), 0.001, 0.999)
    
    # # Fixed, per layer
    # lrs = torch.clamp(hparams['lr_mean'].repeat(num_iters, 1), 0.001, 10.0)
    # mos = torch.clamp(hparams['mo_mean'].repeat(num_iters, 1), 0.001, 0.999)
    
    # Totally learned
    lrs = torch.clamp(hparams['lr_mean'] + hparams['lr_res'], 0.001, 10.0)
    mos = torch.clamp(hparams['mo_mean'] + hparams['mo_res'], 0.001, 0.999)
    
    # --
    # Hyperstep
    
    # if fix_init:
        # set_seeds(seed)
    
    # _ = hopt.zero_grad()
    hlayer = HyperLayer(
        net=net, 
        hparams={
            "lrs"  : lrs,
            "mos"  : mos,
            "meta" : hparams['normal'],
        },
        params=net.parameters(),
        num_iters=num_iters, 
        batch_size=batch_size,
        seed=0 if fix_data else meta_iter,
        verbose=verbose,
    )
    train_hist, val_acc, test_acc = hlayer.run(
        X_train=X_train,
        y_train=y_train, 
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        learn_lrs=False,
        learn_mos=False,
        learn_meta=False,
        # learn_init=False,
        # untrain=True,
    )
    # _ = hopt.step()
    
    # print(F.softmax(hparams['normal'].view(14, 4), dim=-1))
    
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

for lr in to_numpy(lrs).T:
    _ = plt.plot(lr)

show_plot()

