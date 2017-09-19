#!/usr/bin/env python

"""
    run-2.py
    
    !! Need to clean this up so that arguments are passed around sanely
    !! Need to implement own version of `load_data_dicts` and `RandomState`
    !! Need to implement example where meta-parameters get optimized.
        - Could do this by implementing "scaling layer" in nn.Sequential
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

from hypergrad.data import load_data_dicts

from rsub import *
from matplotlib import pyplot as plt

from helpers import to_numpy
from hsgd import HSGD
from hadam import HADAM

np.random.seed(123)
_ = torch.manual_seed(456)
_ = torch.cuda.manual_seed(789)

# --
# IO

batch_size  = 200
num_iters   = 100
N_classes   = 10
N_train     = 10000
N_valid     = 10000
N_tests     = 10000

train_data, valid_data, _ = load_data_dicts(N_train, N_valid, N_tests)

X_train = Variable(torch.FloatTensor(train_data['X'])).cuda()
y_train = Variable(torch.LongTensor(train_data['T'].argmax(axis=1))).cuda()

X_val = Variable(torch.FloatTensor(valid_data['X'])).cuda()
y_val = valid_data['T'].argmax(axis=1)

# --
# Helpers

logit = lambda x: 1 / (1 + (-x).exp())
d_logit = lambda x: x.exp() / ((1 + x.exp()) ** 2) # derivative of logit
d_exp = lambda x: x.exp() # derivative of exponent


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


def deterministic_batch(X, y, sgd_iter, meta_iter, seed=0, batch_size=batch_size):
    rs = np.random.RandomState((seed, meta_iter, sgd_iter))
    idxs = rs.randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    X, y = X[idxs], y[idxs]
    return X, y


def train(net, opt, num_iters, meta_iter, seed=0):
    hist = defaultdict(list)
    for i in tqdm(range(num_iters)):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward(create_graph=True) # !! Have to do this
        
        opt.step(i) if isinstance(opt, HSGD) else opt.step()
        
        train_acc = (to_numpy(net(X_train)).argmax(1) == to_numpy(y_train)).mean()
        val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
        
        hist['train'].append(train_acc)
        hist['val'].append(val_acc)
    
    return opt, hist


def untrain(net, opt, num_iters, meta_iter, seed=0):
    for i in tqdm(range(num_iters)[::-1]):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.zero_grad()
        opt.unstep(lf, i)
    
    return opt


def do_meta_iter(meta_iter, net, lrs, mos):
    opt = HSGD(params=net.parameters(), lrs=lrs, mos=mos)
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, hist = train(net, opt, num_iters=num_iters, meta_iter=meta_iter, seed=0)
    print 'val_acc=%f' % hist['val'][-1]
    
    # Init untrain
    def lf_all():
        return F.cross_entropy(net(X_train), y_train)
    
    opt.zero_grad()
    opt.init_backward(lf_all)
    
    # Untrain
    opt = untrain(net, opt, num_iters, meta_iter)
    
    # Make sure SGD was reversed exactly
    untrained_weights = to_numpy(opt._get_flat_params())
    assert np.all(orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    return opt, hist

# --
# Run

meta_iters = 50
step_size = 0.01

# Initial learning rates -- parameterized as log(lr)
lr_bias = torch.FloatTensor([0.1]).cuda()
lr_res =  torch.FloatTensor(np.full((1, 8), 0.0)).cuda()

mo_bias = torch.FloatTensor([0.5]).cuda()
mo_res = torch.FloatTensor(np.full((1, 8), 0)).cuda()

# Hyper-ADAM optimizer
hyperopt = HADAM([lr_bias, lr_res, mo_bias, mo_res], step_size=step_size)

# Run hypertraining
all_hist = defaultdict(list)
for meta_iter in range(meta_iters):
    print '\nmeta_iter=%d' % meta_iter
    
    net = make_net().cuda()
    opt, hist = do_meta_iter(meta_iter, net, (lr_bias + lr_res).repeat(100, 1), (mo_bias + mo_res).repeat(100, 1))
    
    lr_bias, lr_res, mo_bias, mo_res = hyperopt.step_w_grads([
        opt.d_lrs.sum(dim=0).sum(dim=0),
        opt.d_lrs.sum(dim=0).view(1, -1),
        opt.d_mos.sum(dim=0).sum(dim=0),
        opt.d_mos.sum(dim=0).view(1, -1),
    ])
    
    all_hist['train'].append(hist['train'])
    all_hist['val'].append(hist['val'])
    all_hist['lr_bias'].append(to_numpy(lr_bias))
    all_hist['lr_res'].append(to_numpy(lr_res))
    all_hist['mo_bias'].append(to_numpy(mo_bias))
    all_hist['mo_res'].append(to_numpy(mo_res))
    
    _ = plt.plot(np.hstack(all_hist['train']), alpha=0.25)
    _ = plt.plot(np.hstack(all_hist['val']), alpha=0.25)
    show_plot()


for l in (np.hstack(all_hist['lr_bias']).reshape(-1, 1) + np.vstack(all_hist['lr_res'])).T[::2]:
    _ = plt.plot(l, alpha=0.25)

show_plot()

# Save results
# f = h5py.File('hist-dev.h5')
# f['train'] = np.vstack(all_hist['train'])
# f['test'] = np.vstack(all_hist['val'])
# f.close()

# cis = np.linspace(0, 1, len(f['test'].value))
# for i, l in zip(cis, f['test'].value):
#     _ = plt.plot(l, c=plt.cm.rainbow(i), alpha=0.25)

# show_plot()

# _ = plt.plot(f['test'].value[:,-1])
# show_plot()
