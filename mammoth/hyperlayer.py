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

from .helpers import to_numpy, deterministic_batch
from .hsgd import HSGD

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# --
# Hyperlayer

def safe_backward(p, g):
    if p.grad is not None:
        p.grad.zero_()
    
    p.backward(g)

class HyperLayer(nn.Module):
    def __init__(self, net, params, hparams, 
        num_iters=32, batch_size=32, seed=0, loss_fn=F.cross_entropy, verbose=True):
        
        super().__init__()
        
        self.net        = net
        self.params     = list(params)
        self.hparams    = hparams
        self.num_iters  = num_iters
        self.batch_size = batch_size
        self.seed       = seed
        self.loss_fn    = loss_fn
        self.verbose    = verbose
        
        assert len(self.params) > 0, "HyperLayer.__init__: len(params) == 0"
    
    def run(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None,
        szs=None, untrain=False, check_perfect=True, 
        learn_lrs=True, learn_mos=True, learn_meta=True, learn_init=False):
        """
            learn_weights  : backprop into the initialization
            untrain:       : resulting weights are the same as initial weights
            check_perfect  : make sure weights after untraining are _exactly_ the initial weights
        """
        
        self.opt = HSGD(
            params=self.params,
            hparams=self.hparams,
            szs=szs,
            learn_lrs=learn_lrs,
            learn_mos=learn_mos,
            learn_meta=learn_meta,
        )
        
        if check_perfect:
            orig_weights = self.opt._get_flat_params()
        
        # Run hyperstep
        train_hist = self._train(
            X_train=X_train,
            y_train=y_train,
            num_iters=self.num_iters,
            batch_size=self.batch_size,
        )
        
        # Compute performance
        val_acc  = self._validate(X=X_valid, y=y_valid)
        test_acc = self._validate(X=X_test, y=y_test) if X_test is not None else None
        
        # Save trained state
        state = deepcopy(self.net.state_dict())
        
        # Untrain
        _ = self._untrain(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid, 
            num_iters=self.num_iters,
            batch_size=self.batch_size,
        )
        
        if check_perfect:
            untrained_weights = self.opt._get_flat_params()
            assert (orig_weights == untrained_weights).all(), 'meta_iter: orig_weights != untrained_weights'
        
        if not untrain:
            self.net.load_state_dict(state)
        
        # Update LR, MO, META, initial weights
        if learn_lrs:
            safe_backward(self.hparams['lrs'], self.opt.d_lrs)
        
        if learn_mos:
            safe_backward(self.hparams['mos'], self.opt.d_mos)
        
        if learn_meta:
            safe_backward(self.hparams['meta'], self.opt.d_meta)
        
        if learn_init:
            for p,r in zip(params, self.opt.get_init_params_grad()):
                p.backward(r)
        
        return train_hist, val_acc, test_acc
    
    def _train(self, X_train, y_train, num_iters, batch_size):
        
        # Run forward
        hist = []
        gen = range(num_iters)
        if self.verbose:
            gen = tqdm(gen)
        
        for sgd_iter in gen:
            X_train_batch, y_train_batch = deterministic_batch(X_train, y_train, batch_size, seed=(self.seed, sgd_iter))
            
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
        
        # preds = logits.max(dim=1)[1]
        # acc = (preds == y).float().mean()
        return float(logits.mean())
    
    def _untrain(self, X_train, y_train, X_valid, y_valid, num_iters, batch_size):
        _ = self.opt.zero_grad()
        
        lf_init = lambda: self.loss_fn(self.net(X_valid), y_valid)
        self.opt.init_backward(lf_init)
        
        # Run backward
        gen = range(num_iters)[::-1]
        if self.verbose:
            gen = tqdm(gen)
        
        for sgd_iter in gen:
            X_train_batch, y_train_batch = deterministic_batch(X_train, y_train, batch_size, seed=(self.seed, sgd_iter))
            lf = lambda: self.loss_fn(self.net(X_train_batch), y_train_batch)
            _ = self.opt.zero_grad()
            _ = self.opt.unstep(lf, sgd_iter)
