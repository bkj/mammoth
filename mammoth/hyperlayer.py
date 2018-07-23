#!/usr/bin/env python

"""
    hlayer.py
"""

import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .helpers import to_numpy
from .hsgd import HSGD

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# Helpers

def deterministic_batch(X, y, batch_size, seed):
    idxs = np.random.RandomState(seed).randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    return X[idxs], y[idxs]

# --
# Hyperlayer

class HyperLayer(nn.Module):
    def __init__(self,net, 
        num_iters=32, batch_size=32, seed=0, loss_fn=F.cross_entropy, verbose=True):
    
        super().__init__()
        self.net        = net
        self.num_iters  = num_iters
        self.batch_size = batch_size
        self.seed       = seed
        self.loss_fn    = loss_fn
        self.verbose    = verbose
    
    def run(self, lrs, mos, 
        X_train, y_train, X_valid, y_valid, X_test=None, y_test=None,
        params=None, mts=None, szs=None, update_weights=False, untrain=False, check_perfect=True):
        """
            update_weights : backprop into the initialization
            untrain:       : resulting weights are the same as initial weights
            check_perfect  : make sure weights after untraining are _exactly_ the initial weights
        """
        
        assert X_train.is_cuda and X_valid.is_cuda, "HyperLayer.__init__: not X.is_cuda"
        assert y_train.is_cuda and y_valid.is_cuda, "HyperLayer.__init__: not y.is_cuda"
        
        params = params if params is not None else list(self.net.parameters())
        assert len(params) > 0, "HyperLayer.run: len(params) == 0"
        
        self.opt = HSGD(params=params, lrs=lrs.data, mos=mos.data, szs=szs, mts=mts)
        
        if check_perfect:
            orig_weights = self.opt._get_flat_params()
        
        # Run hyperstep
        train_hist = self._train(
            X_train=X_train,
            y_train=y_train,
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            mts=mts,
        )
        
        # Compute performance
        val_acc = self._validate(X=X_valid, y=y_valid)
        
        if X_test is not None:
            test_acc = self._validate(X=X_test, y=y_test)
        else:
            test_acc = None
        
        # Save trained state
        state = deepcopy(self.net.state_dict())
        
        # Untrain
        self._untrain(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid, 
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            mts=mts,
        )
        
        if check_perfect:
            untrained_weights = self.opt._get_flat_params()
            assert (orig_weights == untrained_weights).all(), 'meta_iter: orig_weights != untrained_weights'
        
        if not untrain:
            self.net.load_state_dict(state)
        
        # Update LR, MO, MTS
        lrs.backward(self.opt.d_lrs)
        mos.backward(self.opt.d_mos)
        if mts is not None:
            mts.backward(self.opt.d_mts)
        
        # Update model parameters
        if update_weights:
            for p,r in zip(params, self.opt.round_dx):
                p.backward(r)
        
        return train_hist, val_acc, test_acc
    
    def _train(self, X_train, y_train, num_iters, batch_size, mts):
        
        # Run forward
        hist = []
        gen = range(num_iters)
        if self.verbose:
            gen = tqdm(gen)
        
        for sgd_iter in gen:
            X_train_batch, y_train_batch = deterministic_batch(X_train, y_train, batch_size, seed=(self.seed, sgd_iter))
            
            _ = self.net.zero_grad()
            _ = self.opt.zero_grad()
            logits = self.net(X_train_batch)
            loss = self.loss_fn(logits, y_train_batch)
            loss.backward()
            _ = self.opt.step(sgd_iter)
            
            hist.append({
                "loss" : float(to_numpy(loss)),
                "acc"  : self._validate(logits=logits, y=y_train_batch)
            })
        
        return hist
    
    def _validate(self, X=None, y=None, logits=None):
        if logits is None:
            logits = self.net(X)
        
        preds = logits.max(dim=1)[1]
        acc = (preds == y).float().mean()
        return float(acc)
    
    def _untrain(self, X_train, y_train, X_valid, y_valid, num_iters, batch_size, mts=None):
        _ = self.opt.zero_grad()
        
        # Initialize backward -- method 1 (not scalable)
        # def lf_all():
        #     return self.loss_fn(self.net(X), y)
        # self.opt.init_backward(lf_all)
        
        # Initialize backward -- method 2 (scalable, less tested)
        for chunk in np.array_split(np.arange(X_valid.size(0)), 10):
            chunk = torch.LongTensor(chunk).cuda()
            loss = self.loss_fn(self.net(X_valid[chunk]), y_valid[chunk], size_average=False)
            loss.backward()
        
        g = self.opt._flatten([p.grad for p in self.opt.params]).data
        g /= X_train.size(0)
        self.opt.d_x = g
        self.opt.g_data = self.opt.d_x.clone()
        self.opt.backward_ready = True
        
        
        # Run backward
        gen = range(num_iters)[::-1]
        if self.verbose:
            gen = tqdm(gen)
        
        for sgd_iter in gen:
            X_train_batch, y_train_batch = deterministic_batch(X_train, y_train, batch_size, seed=(self.seed, sgd_iter))
            
            lf = lambda: self.loss_fn(self.net(X_train_batch), y_train_batch)
            _ = self.opt.zero_grad()
            self.opt.unstep(lf, sgd_iter)
