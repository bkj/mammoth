#!/usr/bin/env python

"""
    hlayer.py
    
    # !! The HyperLayer API isn't the most elegant thing in the world, but it works
    # AFAICT, it'd be most natural to put `_train` in `.forward` and `_untrain` in `_backward_hook`,
    # but when I do that the gradient computation in `opt` just hangs.. I'm guessing that in the
    # background, computation done in those functions is somehow tracked, which I'd want to manually
    # override

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


class HyperLayer(nn.Module):
    def __init__(self, net, lrs, mos, seed=0):
        super(HyperLayer, self).__init__()
        
        self.net = net
        self.lrs = lrs
        self.mos = mos
        self.opt = HSGD(params=net.parameters(), lrs=lrs.data, mos=mos.data)
        
        self.orig_weights = to_numpy(self.opt._get_flat_params())
        self.register_backward_hook(HyperLayer._backward_hook)
        
        self.seed = seed
        
    def _deterministic_batch(self, X, y, batch_size, seed):
        idxs = np.random.RandomState(seed).randint(X.size(0), size=batch_size)
        idxs = torch.LongTensor(idxs).cuda()
        return X[idxs], y[idxs]
    
    def _train(self, X, y, num_iters, batch_size):
        
        # Run forward
        for sgd_iter in tqdm(range(num_iters)):
            Xb, yb = self._deterministic_batch(X, y, batch_size, seed=(self.seed, sgd_iter))
            
            self.opt.zero_grad()
            scores = self.net(Xb)
            loss = F.cross_entropy(scores, yb)
            loss.backward(create_graph=True) # !! Have to do this
            
            self.opt.step(sgd_iter) if isinstance(self.opt, HSGD) else self.opt.step()
    
    def _untrain(self, X, y, num_iters, batch_size):
        
        # Initialize backward
        def lf_all():
            return F.cross_entropy(self.net(X), y)
        
        self.opt.zero_grad()
        self.opt.init_backward(lf_all)
        
        # Run backward
        for sgd_iter in tqdm(range(num_iters)[::-1]):
            Xb, yb = self._deterministic_batch(X, y, batch_size, seed=(self.seed, sgd_iter))
            
            def lf():
                return F.cross_entropy(self.net(Xb), yb)
            
            self.opt.zero_grad()
            self.opt.unstep(lf, sgd_iter)
        
        # Check that backward worked correctly
        untrained_weights = to_numpy(self.opt._get_flat_params())
        assert np.all(self.orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    
    def forward(self, lrs, mos):
        """ hack to get _backward_hook to call w/ correct sized arguments """
        self.tmp = torch.cat([lrs, mos], dim=1)
        return self.tmp
    
    def backward(self):
        """ hack to get _backward_hook to call """
        self.tmp.sum().backward()
    
    @staticmethod
    def _backward_hook(self, grad_input, grad_output):
        """ pass gradients from hypersgd back to lrs/mos """
        return Variable(self.opt.d_lrs), Variable(self.opt.d_mos)

# --
# Run

lrs  = Variable(torch.FloatTensor(np.full((num_iters, 8), 0.3)).cuda(), requires_grad=True)
mos  = Variable(torch.FloatTensor(np.full((num_iters, 8), 0.5)).cuda(), requires_grad=True)
hopt = torch.optim.Adam([lrs, mos], lr=0.05)

for meta_iter in range(50):
    hopt.zero_grad()
    
    # Do forward/reverse round of hypersgd
    net = make_net().cuda()
    h = HyperLayer(net, lrs, mos, seed=meta_iter)
    h._train(X_train, y_train, num_iters, batch_size)
    val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
    print 'val_acc=%f' % val_acc
    h._untrain(X_train, y_train, num_iters, batch_size)
    
    # Compute gradients
    _ = h(lrs, mos)
    h.backward()
    
    # Hypergradient step
    hopt.step()


