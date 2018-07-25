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
    def __init__(self, alpha, layers=[256, 64, 32]):
        super().__init__()
        
        self.alpha = alpha
        self.beta  = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    
    def forward(self, x):
        out = (self.beta[0] ** 2) + (self.alpha[0] ** 2)
        return out.expand(x.shape[0])

# --
# Parameters

num_iters  = 2
batch_size = 4
verbose    = False

# num_iters  = 176 * 10
# batch_size = 256
# verbose    = True

seed       = 345
hyper_lr   = 0.01
init_lr    = 0.01
init_mo    = 0.9
fix_init   = True
fix_data   = True
meta_iters = 100


# --
# Parameterize learning rates + momentum

alpha    = torch.tensor([1.0, 1.0])

n_groups = len(list(Net(alpha=alpha).parameters()))
print('n_groups', n_groups)

# Globally constant
lr_mean  = torch.FloatTensor(np.full((1, 1), init_lr))
mo_mean  = torch.FloatTensor(np.full((1, 1), init_mo))

# # Different per layer
# lr_mean  = torch.FloatTensor(np.full((1, n_groups), init_lr))
# mo_mean  = torch.FloatTensor(np.full((1, n_groups), init_mo))

# # Totally dynamic
# lr_mean  = torch.FloatTensor(np.full((1, n_groups), init_lr))
# mo_mean  = torch.FloatTensor(np.full((1, n_groups), init_mo))
# lr_res   = torch.FloatTensor(np.full((num_iters, n_groups), 0.0))
# mo_res   = torch.FloatTensor(np.full((num_iters, n_groups), 0.0))

# --
# Run

set_seeds(seed)

hparams = {
    "lr_mean" : lr_mean,
    # "lr_res"  : lr_res,
    "mo_mean" : mo_mean,
    # "mo_res"  : mo_res,
    "alpha"   : alpha,
}

for k,v in hparams.items():
    hparams[k] = v.cuda().requires_grad_()

hopt = torch.optim.SGD(
    # params=hparams.values(),
    params=[hparams['alpha']],
    lr=hyper_lr,
    # momentum=0.9,
)

hist = defaultdict(list)
t = time()
for meta_iter in range(0, meta_iters):
    
    # --
    # Transform hyperparameters
    
    # Fixed, same for all layers
    lrs = torch.clamp(hparams['lr_mean'].repeat(num_iters, n_groups), 0.001, 10.0)
    mos = torch.clamp(hparams['mo_mean'].repeat(num_iters, n_groups), 0.001, 0.999)
    
    # # Fixed, per layer
    # lrs = torch.clamp(hparams['lr_mean'].repeat(num_iters, 1), 0.001, 10.0)
    # mos = torch.clamp(hparams['mo_mean'].repeat(num_iters, 1), 0.001, 0.999)
    
    # # Totally learned
    # lrs = torch.clamp(hparams['lr_mean'] + hparams['lr_res'], 0.001, 10.0)
    # mos = torch.clamp(hparams['mo_mean'] + hparams['mo_res'], 0.001, 0.999)
    
    # --
    # Hyperstep
    
    if fix_init:
        set_seeds(seed)
    
    net = Net(alpha=hparams['alpha']).cuda()
    
    _ = hopt.zero_grad()
    hlayer = HyperLayer(
        net=net, 
        hparams={
            "lrs"  : lrs,
            "mos"  : mos,
            "meta" : hparams['alpha'],
        },
        params=net.parameters(),
        num_iters=num_iters, 
        batch_size=batch_size,
        seed=0 if fix_data else meta_iter,
        verbose=verbose,
        loss_fn=F.mse_loss,
    )
    train_hist, val_acc, test_acc = hlayer.run(
        X_train=X_train,
        y_train=y_train.zero_().float(), 
        X_valid=X_valid,
        y_valid=y_valid.zero_().float(),
        X_test=X_test,
        y_test=y_test.zero_().float(),
    )
    _ = hopt.step()
    
    # >>
    # print('alpha', float(hparams['alpha']))
    # print('beta', float(net.beta))
    # print('lr', float(hparams['lr_mean']))
    # <<
    
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
        "time"      : time() - t
    }))
    sys.stdout.flush()

# --
# Plot results

# _ = plt.plot([float(h['alpha']) for h in hist['hparams']], label='alpha')
# _ = plt.plot([float(h['lr_mean'][0][0]) for h in hist['hparams']], label='lr_mean')
# _ = plt.plot([float(h['mo_mean'][0][0]) for h in hist['hparams']], label='mo_mean')
# _ = plt.legend(fontsize=8)
# show_plot()

# _ = plt.plot(hist['val_acc'], label='val_acc')
# _ = plt.plot(hist['test_acc'], label='test_acc')
# _ = plt.legend()
# show_plot()


# # param_names = [k[0] for k in net.named_parameters()]
# # for i, lr in enumerate(to_numpy(mos).T):
# #     if 'weight' in param_names[i]:
# #         _ = plt.plot(lr, label=param_names[i])

# # _ = plt.legend(fontsize=8)
# # show_plot()

