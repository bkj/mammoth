#!/usr/bin/env python

"""
    hyperlayer.py
"""

import os
import sys
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

def zero_grad(p):
    if p is not None:
        if p.grad is not None:
            p.grad.zero_()

def zero_backward(p, g):
    zero_grad(p)
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
    
    def run(self, data,
        learn_lrs=True, learn_mos=True, learn_meta=True, learn_init=False,
        szs=None, untrain=False, check_perfect=True, forward_only=False, mode='one_batch'):
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        if 'X_test' in data:
            X_test = data['X_test']
            y_test = data['y_test']
        
        if learn_init:
            assert untrain, "learn_init and not untrain"
        
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
        test_acc = self._validate(X=X_test, y=y_test) if 'X_test' in data else None
        
        # Save trained state
        if not forward_only:
            state = deepcopy(self.net.state_dict())
            
            # Untrain
            _ = self._untrain(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid, 
                num_iters=self.num_iters,
                batch_size=self.batch_size,
                mode=mode,
            )
            
            # Check that we've returned to _exactly_ correct location
            if check_perfect:
                untrained_weights = self.opt._get_flat_params()
                # assert (orig_weights == untrained_weights).all(), 'meta_iter: orig_weights != untrained_weights'
            
            # Set weights to trained values
            if not untrain:
                self.net.load_state_dict(state)
            
            # Propagate hypergradients
            zero_grad(self.hparams['lrs'])
            zero_grad(self.hparams['mos'])
            zero_grad(self.hparams['meta'])
            
            if learn_lrs:  self.hparams['lrs'].backward(self.opt.d_lrs)
            if learn_mos:  self.hparams['mos'].backward(self.opt.d_mos)
            if learn_meta: self.hparams['meta'].backward(self.opt.d_meta)
            if learn_init: [zero_backward(p, g) for p,g in zip(self.params, self.opt.get_init_params_grad())]
        else:
            print('Hyperlayer.run: forward_only', file=sys.stderr)
        
        return train_hist, val_acc, test_acc
    
    def _train(self, X_train, y_train, num_iters, batch_size):
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
    
    def _validate(self, X=None, y=None, logits=None, mode='one_batch'):
        if logits is None:
            if mode == 'one_batch':
                logits = self.net(X)
                preds = logits.max(dim=1)[1]
                acc = (preds == y).float().mean()
                return float(acc)
            elif mode == 'multi_batch':
                raise Exception
                # correct, total = 0.0, 0.0
                # for Xb, yb in zip(torch.split(X, 10), torch.split(y, 10)):
                #     logits = self.net(Xb)
                #     preds = logits.max(dim=1)[1]
                #     correct += (preds == yb).float().sum()
                #     total += preds.shape[0]
                
                # return float(correct) / total
        else:
            preds = logits.max(dim=1)[1]
            acc = (preds == y).float().mean()
            return float(acc)
    
    def _untrain(self, X_train, y_train, X_valid, y_valid, num_iters, batch_size, mode='one_batch'):
        _ = self.opt.zero_grad()
        
        # One batch, all valid data
        if mode == 'one_batch':
            lf_init = lambda: self.loss_fn(self.net(X_valid), y_valid)
            self.opt.init_backward(lf_init=lf_init)
        else:
            raise Exception
        
        # Random batch
        # idx = torch.randperm(X_valid.shape[0])[:300]
        # idx = torch.arange(300).long()
        # lf_init = lambda: self.loss_fn(self.net(X_valid[idx]), y_valid[idx])
        # self.opt.init_backward(lf_init=lf_init)
        
        # def precompute_lf_init():
        #     n = 0
        #     for Xb, yb in zip(torch.split(X_valid, 10), torch.split(y_valid, 10)):
        #         loss = self.loss_fn(self.net(Xb), yb, size_average=False)
        #         loss.backward()
        #         n += Xb.shape[0]
            
        #     return n
        
        # n = precompute_lf_init()
        # self.opt.init_backward(grad=True, n=n)
        
        # Run backward
        gen = range(num_iters)[::-1]
        if self.verbose:
            gen = tqdm(gen)
        
        for sgd_iter in gen:
            X_train_batch, y_train_batch = deterministic_batch(X_train, y_train, batch_size, seed=(self.seed, sgd_iter))
            lf = lambda: self.loss_fn(self.net(X_train_batch), y_train_batch)
            _ = self.opt.zero_grad()
            _ = self.opt.unstep(lf, sgd_iter)
