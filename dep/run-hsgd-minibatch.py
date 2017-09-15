#!/usr/bin/env python

"""
    pytorch-hypergrad-simple.py
"""

import numpy as np

import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from torch import autograd
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from rsub import *
from matplotlib import pyplot as plt

from hsgd import HSGD

np.random.seed(456)

def loss_fun(X, y):
    return ((X - y) ** 2).mean()

num_iters = 50
batch_size = 10
input_dim  = 10

lrs        = 0.1 + np.zeros(num_iters)
momentums  = 0.9 + np.zeros(num_iters)

truth = torch.rand((10, 5))
X = torch.LongTensor(np.random.choice(10, 1000))
y = truth[X]

data = Variable(X)
y = Variable(y).double()

all_loss = []
for meta_epoch in range(25):
    # Initialize model
    
    l = nn.Embedding(10, 5).double()
    orig = l.weight.data.numpy().copy()
    
    # sgd
    loss_hist = []
    opt = HSGD(l.parameters(), lrs, momentums, num_iters=num_iters)
    for i in range(num_iters):
        np.random.seed((123, meta_epoch, i))
        batch = torch.LongTensor(np.random.choice(data.size(0), batch_size))
        opt.zero_grad()
        pred = l(data[batch])
        loss = loss_fun(pred, y[batch])
        loss.backward()
        opt.step(i)
        
        loss_hist.append(loss_fun(l(data), y).data[0])
    
    all_loss.append(loss_hist)
    
    # Trained weights
    trained = l.weight.data.numpy().copy()
    
    # hypersgd        
    for i in range(num_iters)[::-1]:
        # np.random.seed((123, meta_epoch, i))
        # batch = torch.LongTensor(np.random.choice(data.size(0), batch_size))
        def lf():
            return loss_fun(l(data), y)
            
        opt.unstep(lf, i)
        
    untrained = l.weight.data.numpy().copy()
    assert np.all(untrained == orig)
    
    lrs -= opt.d_lrs.numpy()
    momentums -= opt.d_momentums.numpy()
    print all_loss[-1][-1]

