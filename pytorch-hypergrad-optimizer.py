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

# --
# Helpers

class HSGD(Optimizer):
    # """ stripped down torch default, but w/ signs flipped to match hypergrad """
    def __init__(self, params, lrs, momentums, num_epochs):
        super(HSGD, self).__init__(params, {
            "lrs" : lrs,
            "momentums" : momentums,
        })
        self.d_lrs       = torch.zeros(num_epochs).double()
        self.d_momentums = torch.zeros(num_epochs).double()
        
    def step(self, i):
        for group in self.param_groups:
            momentum = torch.DoubleTensor([group['momentums'][i]])
            lr = torch.DoubleTensor([group['lrs'][i]])
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                g = param.grad.data
                
                param_state = self.state[param]
                
                if 'X' not in param_state:
                    param_state['X'] = ETensor(param.data.clone())
                    
                if 'V' not in param_state:
                    param_state['V'] = ETensor(g.clone().zero_())
                
                _ = param_state['V'].mul(momentum).sub(g)
                _ = param_state['X'].add(lr * param_state['V'].val)
                param.data.set_(param_state['X'].val)
    
    def unstep(self, loss_fun, i=0):
        for group in self.param_groups:
            
            momentum = torch.DoubleTensor([group['momentums'][i]])
            lr = torch.DoubleTensor([group['lrs'][i]])
            
            # Update parameters in all layers
            for param in group['params']:
                param_state = self.state[param]
                
                if 'd_x' not in param_state:
                    param_state['d_x'] = autograd.grad(loss_fun(), param)[0].data
                
                if 'grad_proj_x' not in param_state:
                    param_state['grad_proj_x'] = lambda x, d: (autograd.grad(loss_fun(), x, create_graph=True)[0] * d).sum()
                
                if 'd_v' not in param_state:
                    param_state['d_v'] = torch.zeros(param.size()).double()
                
                self.d_lrs[i] += (param_state['d_x'] * param_state['V'].val).sum()
                
                _ = param_state['X'].sub(lr * param_state['V'].val)
                param.data.set_(param_state['X'].val)
            
            # Update velocities in all layers
            for param in group['params']:
                param_state = self.state[param]
                g = autograd.grad(loss_fun(), param)[0].data
                _ = param_state['V'].add(g).div(momentum)
                
                param_state['d_v'] += param_state['d_x'] * lr
                
                self.d_momentums[i] += (param_state['d_v'] * param_state['V'].val).sum()
                
                d_vpar = Parameter(param_state['d_v'])
                param_state['d_x'] -= autograd.grad(param_state['grad_proj_x'](param, d_vpar), param)[0].data
                
                param_state['d_v'] = param_state['d_v'] * momentum


# --

np.random.seed(456)

def loss_fun(X, y):
    return ((X - y) ** 2).mean()

num_epochs = 20
input_dim = 6

lrs        = 0.1 + np.zeros(num_epochs)
momentums  = 0.9 + np.zeros(num_epochs)

data = Variable(torch.ones((1, 1))).double()
y    = Variable(torch.DoubleTensor(np.random.uniform(0, 1, input_dim))).view(1, -1)

all_loss = []
for _ in range(25):
    # Initialize model
    
    # l = nn.Linear(1, input_dim, bias=False).double()
    # _ = l.weight.data.zero_()
    # _ = l.weight.data.add_(0.25)
    # orig = l.weight.data.numpy().copy()
    
    l = nn.Sequential(
        # nn.Linear(1, 1, bias=False),
        nn.Linear(1, input_dim, bias=False),
    ).double()
    
    # sgd
    loss_hist = []
    opt = HSGD(l.parameters(), lrs, momentums, num_epochs=num_epochs)
    for i in range(num_epochs):
        opt.zero_grad()
        pred = l(data)
        loss = loss_fun(pred, y)
        loss.backward()
        opt.step(i)
        
        loss_hist.append(loss_fun(l(data), y).data[0])
    
    all_loss.append(loss_hist)
    
    # Trained weights
    # trained = l.weight.data.numpy().copy()
    
    # hypersgd
    def lf():
        return loss_fun(l(data), y)
        
    for i in range(num_epochs)[::-1]:
        opt.unstep(lf, i)
        
    # untrained = l.weight.data.numpy().copy()
    # assert np.all(untrained == orig)
    
    lrs -= 0.1 * opt.d_lrs.numpy()
    momentums -= 0.1 * opt.d_momentums.numpy()
    print all_loss[-1][-1]
