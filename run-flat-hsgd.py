#!/usr/bin/env python

"""
    run-flat-hsgd.py
"""

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

from helpers import to_numpy
from flat_hsgd import FlatHSGD

# --
# IO

batch_size  = 200
num_iters   = 100
N_classes   = 10
N_train     = 10000
N_valid     = 10000
N_tests     = 10000

train_data, valid_data, _ = load_data_dicts(N_train, N_valid, N_tests)

X_train = Variable(torch.DoubleTensor(train_data['X'])).cuda()
y_train = Variable(torch.LongTensor(train_data['T'].argmax(axis=1))).cuda()

X_val = Variable(torch.DoubleTensor(valid_data['X'])).cuda()
y_val = valid_data['T'].argmax(axis=1)

# --
# Helpers

logit = lambda x: 1 / (1 + (-x).exp())

def make_net(weight_scale=np.exp(-3), layers=[50, 50, 50]):
    
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
            _ = child.weight.data.normal_(0, weight_scale)
    
    net = net.double()
    return net


def deterministic_batch(X, y, seed, batch_size=batch_size):
    np.random.seed(seed)
    sel = torch.LongTensor(np.random.choice(X.size(0), batch_size)).cuda()
    X, y = X[sel], y[sel]
    return X, y


def train(net, opt, num_iters, seed=0):
    train_hist, val_hist = [], []
    for i in tqdm(range(num_iters)):
        X, y = deterministic_batch(X_train, y_train, seed=(seed, i))
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        
        opt.step(i) if isinstance(opt, FlatHSGD) else opt.step()
        
        train_acc = (to_numpy(net(X_train)).argmax(1) == to_numpy(y_train)).mean()
        train_hist.append(train_acc)
        
        val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
        val_hist.append(val_acc)
    
    return opt, val_hist, train_hist


def untrain(net, opt, num_iters, seed=0):
    for i in tqdm(range(num_iters)[::-1]):
        X, y = deterministic_batch(X_train, y_train, seed=(seed, i))
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.unstep(lf, i)
    
    return opt


def meta_iter(meta_epoch, seed=None):
    if seed:
        _ = torch.manual_seed(seed)
        _ = torch.cuda.manual_seed(seed)
    
    net = make_net().cuda()
    
    opt = FlatHSGD(net.parameters(),
        lrs=lrs.exp(),
        momentums=logit(momentums),
        cuda=True
    )
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, val_hist, train_hist = train(net, opt, num_iters, seed=meta_epoch)
    trained_weights = to_numpy(opt._get_flat_params())
    print {
        "train_acc" : train_hist[-1],
        "val_acc" : val_hist[-1]
    }
    
    # Untrain
    opt = untrain(net, opt, num_iters, seed=meta_epoch)
    untrained_weights = to_numpy(opt._get_flat_params())
    assert np.all(orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    return opt, val_hist, train_hist

# --
# Run

meta_epochs = 100

lrs = torch.DoubleTensor(np.full((num_iters, 8), -1.0)).cuda()
momentums = torch.DoubleTensor(np.full((num_iters, 8), 0.0)).cuda()

b1 = 0.1
b2 = 0.01
eps = 10 ** -4
lam = 10 ** -4
step_size = 0.02

m = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(momentums.size()).double().cuda()]
v = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(momentums.size()).double().cuda()]

all_val_hists, all_train_hists = [], []
for meta_epoch in range(meta_epochs):
    print "\nmeta_epoch=%d" % meta_epoch
    
    opt, val_hist, train_hist = meta_iter(meta_epoch, seed=None)
    all_train_hists.append(train_hist)
    all_val_hists.append(val_hist)
    
    # ADAM step -- need to apply to all hypergrads
    b1t = 1 - (1 - b1) * (lam ** meta_epoch)
    
    g = opt.d_lrs * lrs.exp() # !! Is this right
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
