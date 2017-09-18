#!/usr/bin/env python

"""
    run.py
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
from hypergrad.util import RandomState

from rsub import *
from matplotlib import pyplot as plt

from helpers import to_numpy
from hsgd import HSGD

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

X_train = Variable(torch.DoubleTensor(train_data['X'])).cuda()
y_train = Variable(torch.LongTensor(train_data['T'].argmax(axis=1))).cuda()

X_val = Variable(torch.DoubleTensor(valid_data['X'])).cuda()
y_val = valid_data['T'].argmax(axis=1)

# --
# Helpers

logit = lambda x: 1 / (1 + (-x).exp())
d_logit = lambda x: x.exp() / ((1 + x.exp()) ** 2) # derivative of logit
d_exp = lambda x: x.exp() # derivative of exponent

def set_net_weights(net, val):
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        if len(p.size()) == 1:
            p.data.set_(val[offset:offset + numel].view_as(p.data))
        else:
            p.data.set_(val[offset:offset + numel].view_as(p.data.t()).t()) # !! autograd ordering
        offset += numel
    print offset

def make_net(weight_scale=np.exp(-3), layers=[50, 50, 50]):
    
    net = nn.Sequential(
        nn.Linear(784, layers[0]),
        nn.Tanh(),
        nn.Linear(layers[0], layers[1]),
        nn.Tanh(),
        nn.Linear(layers[1], layers[2]),
        nn.Tanh(),
        nn.Linear(layers[2], 10),
        nn.LogSoftmax(),
    )
    
    for child in net.children():
        if isinstance(child, nn.Linear):
            _ = child.weight.data.normal_(0, weight_scale)
    
    net = net.double()
    return net


def deterministic_batch(X, y, sgd_iter, meta_iter, seed=0, batch_size=batch_size):
    rs = RandomState((seed, meta_iter, sgd_iter))
    idxs = rs.randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    X, y = X[idxs], y[idxs]
    return X, y


def train(net, opt, num_iters, meta_iter, seed=0):
    train_hist, val_hist = [], []
    for i in tqdm(range(num_iters)):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        
        opt.step(i) if isinstance(opt, HSGD) else opt.step()
        
        train_acc = (to_numpy(net(X_train)).argmax(1) == to_numpy(y_train)).mean()
        train_hist.append(train_acc)
        
        val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
        val_hist.append(val_acc)
    
    return opt, val_hist, train_hist


def untrain(net, opt, num_iters, meta_iter, seed=0):
    for i in tqdm(range(num_iters)[::-1]):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.unstep(lf, i)
    
    return opt


def do_meta_iter(meta_iter, net, lrs, mos):
    opt = HSGD(
        params=net.parameters(),
        lrs=lrs.exp(),
        mos=logit(mos),
    )
    
    # opt = torch.optim.SGD(
    #     net.parameters(),
    #     lr=0.1,
    #     momentum=0.5,
    # )
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, val_hist, train_hist = train(net, opt, num_iters=num_iters, meta_iter=meta_iter, seed=0)
    trained_weights = to_numpy(opt._get_flat_params())
    print {"train_acc" : train_hist[-1], "val_acc" : val_hist[-1]}
    
    # Init untrain
    def lf_all():
        return F.cross_entropy(net(X_train), y_train)
    
    opt.init_backward(lf_all)
    
    # Untrain
    opt = untrain(net, opt, num_iters, meta_iter)
    untrained_weights = to_numpy(opt._get_flat_params())
    assert np.all(orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    return opt, val_hist, train_hist

# --
# Run


lrs = torch.DoubleTensor(np.full((num_iters, 8), -1.0)).cuda()
mos = torch.DoubleTensor(np.full((num_iters, 8), 0.0)).cuda()

meta_iters = 50

b1 = 0.1
b2 = 0.01
eps = 10 ** -4
lam = 10 ** -4
step_size = 0.04

m = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(mos.size()).double().cuda()]
v = [torch.zeros(lrs.size()).double().cuda(), torch.zeros(mos.size()).double().cuda()]

all_val_hists, all_train_hists = [], []
for meta_iter in range(meta_iters):
    print "\nmeta_iter=%d" % meta_iter
    
    net = make_net(weight_scale=np.exp(-3), layers=[50, 50, 50]).cuda()
    opt, val_hist, train_hist = do_meta_iter(meta_iter, net, lrs, mos)
    all_train_hists.append(train_hist)
    all_val_hists.append(val_hist)
    
    # ADAM step -- need to apply to all hypergrads
    b1t = 1 - (1 - b1) * (lam ** meta_iter)
    
    g = opt.d_lrs * d_exp(lrs) # !!
    m[0] = b1t * g + (1-b1t) * m[0]
    v[0] = b2 * (g ** 2) + (1 - b2) * v[0]
    mhat = m[0] / (1 - (1 - b1) ** (meta_iter + 1))
    vhat = v[0] / (1 - (1 - b2) ** (meta_iter + 1))
    lrs -= step_size * mhat / (vhat.sqrt() + eps)
    
    g = opt.d_mos * d_logit(mos) # !!
    m[1] = b1t * g + (1-b1t) * m[1]
    v[1] = b2 * (g ** 2) + (1 - b2) * v[1]
    mhat = m[1] / (1 - (1 - b1) ** (meta_iter + 1))
    vhat = v[1] / (1 - (1 - b2) ** (meta_iter + 1))
    mos -= step_size * mhat / (vhat.sqrt() + eps)


cm = np.linspace(0, 1, len(all_val_hists))
for i, v in enumerate(all_val_hists):
    _ = plt.plot(v, c=plt.cm.rainbow(cm[i]), alpha=0.25)

show_plot()

