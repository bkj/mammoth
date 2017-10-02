#!/usr/bin/env python

"""
    hlayer.py
"""

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy
from hsgd import HSGD

if torch.__version__ != '0.2.0+9b8f5eb_dev':
    raise Exception("torch.__version__ != '0.2.0+9b8f5eb_dev'")
    os._exit(1)
else:
    # !! Ordinarily forces use of best algorithm, but hacked to use default (determnistic) ops
    torch.backends.cudnn.benchmark = True


class HyperLayer(nn.Module):
    def __init__(self, X, y, num_iters, batch_size, seed=0):
        super(HyperLayer, self).__init__()
        
        self.X = X
        self.y = y
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.seed = seed
        
        self.register_backward_hook(HyperLayer._backward_hook)
    
    def __call__(self, net, lrs, mos, val_data=None, szs=None, cheap=False):
        self.net = net
        self.lrs = lrs
        self.mos = mos
        self.opt = HSGD(params=net.parameters(), lrs=lrs.data, mos=mos.data, szs=szs)
        
        self.orig_weights = to_numpy(self.opt._get_flat_params())
        
        # Run hyperstep
        self.loss_hist, self.acc_hist = self._train(self.X, self.y, self.num_iters, self.batch_size)
        self.val_acc = self._validate(*val_data) if val_data else None
        
        state = copy.deepcopy(self.net.state_dict())
        self._untrain(self.X, self.y, self.num_iters, self.batch_size, cheap=cheap)
        self.net.load_state_dict(state)
        
        # Return dummy loss, so we can propagate errors
        tmp = super(HyperLayer, self).__call__(self.lrs, self.mos)
        return tmp.sum()
        
    def _deterministic_batch(self, X, y, batch_size, seed):
        idxs = np.random.RandomState(seed).randint(X.size(0), size=batch_size)
        idxs = torch.LongTensor(idxs).cuda()
        return X[idxs], y[idxs]
    
    def _train(self, X, y, num_iters, batch_size):
        
        # Run forward
        loss_hist, acc_hist = [], []
        for sgd_iter in tqdm(range(num_iters)):
            Xb, yb = self._deterministic_batch(X, y, batch_size, seed=(self.seed, sgd_iter))
            
            self.opt.zero_grad()
            scores = self.net(Xb)
            loss = F.cross_entropy(scores, yb)
            loss.backward(create_graph=True) # !! Have to do this
            
            self.opt.step(sgd_iter) if isinstance(self.opt, HSGD) else self.opt.step()
            
            loss_hist.append(to_numpy(loss))
            acc_hist.append(self._validate(scores=scores, y=yb))
        
        return np.hstack(loss_hist), np.hstack(acc_hist)
    
    def _validate(self, X=None, y=None, scores=None):
        if scores is None:
            scores = self.net(X)
        
        preds = to_numpy(scores).argmax(1)
        act = to_numpy(y)
        return (preds == act).mean()
    
    def _untrain(self, X, y, num_iters, batch_size, cheap=False):
        self.opt.zero_grad()
        
        try:
            # Initialize backward -- method 1 (not scalable)
            def lf_all():
                return F.cross_entropy(self.net(X), y)
            
            self.opt.init_backward(lf_all)
        except:
            # Initialize backward -- method 2 (scalable, less tested)
            for chunk in np.array_split(np.arange(X.size(0)), 10):
                chunk = torch.LongTensor(chunk).cuda()
                loss = F.cross_entropy(self.net(X[chunk]), y[chunk], size_average=False)
                loss.backward()
            
            g = self.opt._flatten([p.grad for p in self.opt.params]).data
            g /= X.size(0)
            self.opt.d_x = g
            self.opt.g_data = self.opt.d_x.clone()
        
        self.opt.backward_ready = True
        
        # Run backward
        for sgd_iter in tqdm(range(num_iters)[::-1]):
            Xb, yb = self._deterministic_batch(X, y, batch_size, seed=(self.seed, sgd_iter))
            
            def lf():
                return F.cross_entropy(self.net(Xb), yb)
            
            self.opt.zero_grad()
            if cheap:
                self.opt.unstep_cheap(lf, sgd_iter, one_step=False)
            else:
                self.opt.unstep(lf, sgd_iter)
            
        
        # Check that backward worked correctly
        untrained_weights = to_numpy(self.opt._get_flat_params())
        assert np.all(self.orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    def forward(self, lrs, mos):
        """ hack to get _backward_hook to call w/ correct sized arguments """
        return torch.cat([lrs, mos], dim=1)
    
    @staticmethod
    def _backward_hook(self, grad_input, grad_output):
        """ pass gradients from hypersgd back to lrs/mos """
        return Variable(self.opt.d_lrs), Variable(self.opt.d_mos)

