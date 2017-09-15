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


def make_net(layers=[50, 50, 50]):
    
    return nn.Sequential(
        nn.Linear(784, layers[0]),
        nn.Tanh(),
        nn.Linear(layers[0], layers[1]),
        nn.Tanh(),
        nn.Linear(layers[1], layers[2]),
        nn.Tanh(),
        nn.Linear(layers[2], 10)
    ).double()
    

# --
# Train

batch_size  = 200
N_iters     = 100
N_classes   = 10
N_train     = 10000
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


logit = lambda x: 1 / (1 + (-x).exp())

def meta_iter(meta_epoch, seed=None):
    if seed:
        _ = torch.manual_seed(seed)
        _ = torch.cuda.manual_seed(seed)
    
    net = make_net().cuda()
    
    for child in net.children():
        if isinstance(child, nn.Linear):
            _ = child.weight.data.normal_(0, np.exp(-3))
    
    opt = FlatHSGD(net.parameters(), 
        lrs=lrs.exp(), 
        momentums=logit(momentums), 
        cuda=True
    )
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, val_hist = train(net, opt, N_iters, seed=meta_epoch)
    trained_weights = to_numpy(opt._get_flat_params())
    print 'final acc=%f' % val_hist[-1]
    
    # Untrain
    opt = untrain(net, opt, N_iters, seed=meta_epoch)
    untrained_weights = to_numpy(opt._get_flat_params())
    assert(np.all(orig_weights == untrained_weights))
    
    return opt, val_hist

# --
# Run

meta_epochs = 100

# !! Should be parameterized on the log scale
lrs = torch.DoubleTensor(np.full((N_iters, 8), -1)).cuda()
momentums = torch.DoubleTensor(np.full((N_iters, 8), 0)).cuda()

b1 = 0.1
b2 = 0.01
eps = 10**-4
lam = 10**-4
step_size = 0.05

m = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(momentums.size()).double().cuda()]
v = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(momentums.size()).double().cuda()]

all_hists = []
for meta_epoch in range(meta_epochs):
    
    b1t = 1 - (1 - b1) * (lam ** meta_epoch)
    opt, val_hist = meta_iter(meta_epoch, seed=None)
    all_hists.append(val_hist)
    
    g = opt.d_lrs * lrs.exp() # !! Is this right?
    m[0] = b1t * g + (1-b1t) * m[0]
    v[0] = b2 * (g ** 2) + (1 - b2) * v[0]
    mhat = m[0] / (1 - (1 - b1) ** (meta_epoch + 1))
    vhat = v[0] / (1 - (1 - b2) ** (meta_epoch + 1))
    lrs -= step_size * mhat / (vhat.sqrt() + eps)
    
    g = opt.d_momentums * momentums.exp() / ((1 + momentums.exp()) ** 2) # !! Is this right?
    m[1] = b1t * g + (1-b1t) * m[0]
    v[1] = b2 * (g ** 2) + (1 - b2) * v[1]
    mhat = m[1] / (1 - (1 - b1) ** (meta_epoch + 1))
    vhat = v[1] / (1 - (1 - b2) ** (meta_epoch + 1))
    momentums -= step_size * mhat / (vhat.sqrt() + eps)
    
    # for l in to_numpy(momentums[:,::2]).T:
    #     _ = plt.plot(1 / (1 + np.exp(-l)), alpha=0.5)
    for l in to_numpy(lrs[:,::2]).T:
        _ = plt.plot(np.exp(l), alpha=0.5)
        
    show_plot()
    
    for h in all_hists:
        _ = plt.plot(h, alpha=0.5)
        
    show_plot()
