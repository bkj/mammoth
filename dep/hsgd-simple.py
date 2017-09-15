#!/usr/bin/env python

"""
    pytorch-hypergrad-simple.py
"""

import numpy as np

import torch
from torch import optim
from torch import autograd
from torch import nn`
from torch.autograd import Variable
from torch.nn import Parameter

np.random.seed(456)

# --
# Helpers

def loss_fun(X, y):
    return ((X - y) ** 2).mean()


def do_sgd(X, y, lrs, momentums):
    V = None
    loss_hist = []
    for i in range(num_epochs):
        lr = torch.DoubleTensor([lrs[i]])
        momentum = torch.DoubleTensor([momentums[i]])
        
        Xpar = Parameter(X.val)
        g = autograd.grad(loss_fun(Xpar, y), Xpar, only_inputs=True)[0].data
        
        if not V:
            V = ETensor(g.clone().zero_())
        
        _ = V.mul(momentum).sub(g)
        _ = X.add(lr * V.val)
        
        loss_hist.append(loss_fun(Parameter(X.val), y).data[0])
    
    return X, V, loss_hist


def do_hypersgd(X, V, lrs, momentums):
    d_lrs       = torch.zeros(num_epochs).double()
    d_momentums = torch.zeros(num_epochs).double()
    d_v         = torch.zeros(X.size).double()
    
    # Gradient w.r.t. model weights
    Xpar = Parameter(X.val)
    d_x = autograd.grad(loss_fun(Xpar, y), Xpar)[0].data
    
    grad_proj_x = lambda x, d: (autograd.grad(loss_fun(x, y), x, create_graph=True)[0] * d).sum()
    
    for i in range(num_epochs)[::-1]:
        lr = torch.DoubleTensor([lrs[i]])
        momentum = torch.DoubleTensor([momentums[i]])
        
        # Update learning rates
        d_lrs[i] = (d_x * V.val).sum()
        
        # Exactly reverse SGD
        _ = X.sub(lr * V.val)
        Xpar = Parameter(X.val)
        g = autograd.grad(loss_fun(Xpar, y), Xpar)[0].data
        _ = V.add(g).div(momentum)
        
        d_v += d_x * lr
        
        # Update momentum
        d_momentums[i] = (d_v * V.val).sum()
        
        # Update weights
        d_vpar = Parameter(d_v)
        d_x -= autograd.grad(grad_proj_x(Xpar, d_vpar), Xpar)[0].data
        
        d_v = d_v * momentum
        
    return X, V, d_lrs, d_momentums

# --
# Create data

np.random.seed(456)
dim = 6
num_epochs = 20

lrs = 0.1 + np.zeros(num_epochs)
momentums = 0.9 + np.zeros(num_epochs)

X0 = np.full((dim, ), 0.25)
y0 = np.random.uniform(0, 1, dim)

all_loss = []
for _ in range(25):
    X = ETensor(torch.DoubleTensor(X0.copy()))
    y = Parameter(torch.DoubleTensor(y0))
    
    # sgd
    X, V, loss_hist = do_sgd(X, y, lrs, momentums)
    Xn = X.val.numpy().copy()
    all_loss.append(loss_hist)
    
    # hypersgd
    X, V, d_lrs, d_momentums = do_hypersgd(X, V, lrs, momentums)
    assert np.all(X.val.numpy() == X0)
    
    lrs -= d_lrs.numpy()
    print all_loss[-1][-1]
