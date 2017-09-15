#!/usr/bin/env python

import sys
sys.path.append('/home/bjohnson/software/autograd/')
sys.path.append('/home/bjohnson/software/hypergrad')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from hypergrad.data import load_data_dicts

from rsub import *
from matplotlib import pyplot as plt

# --
# Run


def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def make_net(layers=[25, 25, 25]):
    
    net = nn.Sequential(
        nn.Linear(784, layers[0]),
        nn.Tanh(),
        nn.Linear(layers[0], layers[1]),
        nn.Tanh(),
        nn.Linear(layers[1], layers[2]),
        nn.Tanh(),
        nn.Linear(layers[2], 10)
    )
    
    for child in net.children():
        if isinstance(child, nn.Linear):
            _ = child.weight.data.normal_(0, np.exp(-3))
            
    net = net.double()
    return net

# --
# Train

batch_size  = 128
# N_iters     = 1500
N_iters     = 250
N_classes   = 10
N_train     = 50000
N_valid     = 10000
N_tests     = 10000

train_data, valid_data, _ = load_data_dicts(N_train, N_valid, N_tests)

X_train = train_data['X']
y_train = train_data['T']

X_val = Variable(torch.DoubleTensor(valid_data['X'])).cuda()
y_val = valid_data['T'].argmax(axis=1)

def deterministic_batch(X, y, seed, batch_size=batch_size):
    np.random.seed(seed)
    sel = np.random.choice(X.shape[0], batch_size)
    X, y = X[sel], y[sel].argmax(axis=1)
    X, y = Variable(torch.DoubleTensor(X)).cuda(), Variable(torch.LongTensor(y)).cuda()
    return X, y

def train(net, opt, N_iters, seed=0):
    hist, val_hist = [], []
    for i in tqdm(range(N_iters)):
        X, y = deterministic_batch(X_train, y_train, (seed, i))
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        
        if isinstance(opt, FlatHSGD) or isinstance(opt, HSGD):
            opt.step(i)
        else:
            opt.step()
        
        val_hist.append((to_numpy(net(X_val)).argmax(1) == y_val).mean())
    
    return opt, val_hist


def untrain(net, opt, N_iters, seed=0):
    for i in tqdm(range(N_iters)[::-1]):
        X, y = deterministic_batch(X_train, y_train, (seed, i))
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.unstep(lf, i)
    
    return opt

# --
# Run

# lrs = torch.DoubleTensor(np.full((N_iters, 8), 0.3)).cuda()
# momentums = torch.DoubleTensor(np.full((N_iters, 8), 0.5)).cuda()

lrs = np.full(N_iters, 0.3)
momentums = np.full(N_iters, 0.5)

all_hists = []
for meta_epoch in range(10):
    _ = torch.manual_seed(234)
    _ = torch.cuda.manual_seed(234)
    
    net = make_net().cuda()
    
    opt = FlatHSGD(
        net.parameters(),
        lrs=lrs,
        momentums=momentums,
        num_iters=N_iters,
        cuda=True
    )
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, val_hist = train(net, opt, N_iters, seed=meta_epoch)
    trained_weights = to_numpy(opt._get_flat_params())
    print 'final acc=%f' % val_hist[-1]
    all_hists.append(val_hist)
    
    # Untrain
    opt = untrain(net, opt, N_iters, seed=meta_epoch)
    untrained_weights = to_numpy(opt._get_flat_params())
    assert(np.all(orig_weights == untrained_weights))
    
    # Update hyperparameters
    lrs -= 0.1 * to_numpy(opt.d_lrs)
    momentums -= 0.1 * to_numpy(opt.d_momentums)


for h in np.vstack(all_hists):
    _ = plt.plot(h, alpha=0.25)

show_plot()


_ = plt.plot(lrs)
show_plot()


opt = FlatHSGD(
    net.parameters(), 
    lrs=lrs,
    momentums=momentums, 
    num_iters=N_iters, 
    cuda=True
)
z = torch.FloatTensor([0.1] * len(opt._params))
opt._fill_parser(z)

