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

from helpers import to_numpy
from hsgd import HSGD

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class HyperLayer(nn.Module):
    def __init__(self, X, y, num_iters, batch_size, seed=0, loss_function=F.cross_entropy):
        super(HyperLayer, self).__init__()
        
        assert X.is_cuda, "HyperLayer.__init__: not X.is_cuda"
        assert y.is_cuda, "HyperLayer.__init__: not y.is_cuda"
        
        self.X = X
        self.y = y
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.seed = seed
        self.loss_function = loss_function
    
    def __call__(self, net, lrs, mos, params=None, mts=None, val_data=None, szs=None, cheap=False):
        self.net = net
        
        params = params if params is not None else list(net.parameters())
        assert len(params) > 0, "HyperLayer.__call__: len(params) == 0"
        
        self.opt = HSGD(params=params, lrs=lrs.data, mos=mos.data, szs=szs, mts=mts)
        
        self.orig_weights = to_numpy(self.opt._get_flat_params())
        
        # Run hyperstep
        self.loss_hist, self.acc_hist = self._train(self.X, self.y, self.num_iters, self.batch_size, mts)
        self.val_acc = self._validate(*val_data) if val_data else None
        
        state = copy.deepcopy(self.net.state_dict())
        self._untrain(self.X, self.y, self.num_iters, self.batch_size, cheap=cheap)
        self.net.load_state_dict(state)
        
        lrs.backward(self.opt.d_lrs)
        mos.backward(self.opt.d_mos)
        if mts is not None:
            mts.backward(self.opt.d_mts)
        
    def _deterministic_batch(self, X, y, batch_size, seed):
        idxs = np.random.RandomState(seed).randint(X.size(0), size=batch_size)
        idxs = torch.LongTensor(idxs).cuda()
        return X[idxs], y[idxs]
    
    def _train(self, X, y, num_iters, batch_size, mts):
        
        # Run forward
        loss_hist, acc_hist = [], []
        for sgd_iter in tqdm(range(num_iters)):
            Xb, yb = self._deterministic_batch(X, y, batch_size, seed=(self.seed, sgd_iter))
            
            self.net.zero_grad()
            self.opt.zero_grad()
            scores = self.net(Xb)
            loss = 1.0 * self.loss_function(scores, yb)
            loss.backward()
            
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
        # return ((preds - act) ** 2).sum()
    
    def _untrain(self, X, y, num_iters, batch_size, cheap=False):
        self.opt.zero_grad()
        
        # Initialize backward -- method 1 (not scalable)
        # def lf_all():
        #     return self.loss_function(self.net(X), y)
        
        # self.opt.init_backward(lf_all)
        
        # Initialize backward -- method 2 (scalable, less tested)
        for chunk in np.array_split(np.arange(X.size(0)), 10):
            chunk = torch.LongTensor(chunk).cuda()
            loss = self.loss_function(self.net(X[chunk]), y[chunk], size_average=False)
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
                return self.loss_function(self.net(Xb), yb)
            
            self.opt.zero_grad()
            if cheap:
                self.opt.unstep_cheap(lf, sgd_iter, one_step=False)
            else:
                self.opt.unstep(lf, sgd_iter)
        
        # Check that backward worked correctly
        untrained_weights = to_numpy(self.opt._get_flat_params())
        assert np.all(self.orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'

