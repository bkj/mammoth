#!/usr/bin/env python

"""
    hlayer.py
"""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy
from hsgd import HSGD

class HyperLayer(nn.Module):
    def __init__(self, X, y, num_iters, batch_size, seed=0):
        super(HyperLayer, self).__init__()
        
        self.X = X
        self.y = y
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.seed = seed
        
        self.register_backward_hook(HyperLayer._backward_hook)
    
    def __call__(self, net, lrs, mos, val_data=None):
        self.net = net
        self.lrs = lrs
        self.mos = mos
        self.opt = HSGD(params=net.parameters(), lrs=lrs.data, mos=mos.data)
        self.orig_weights = to_numpy(self.opt._get_flat_params())
        
        # Run hyperstep
        self._train(self.X, self.y, self.num_iters, self.batch_size)
        if val_data:
            self.val_acc = (to_numpy(net(val_data[0])).argmax(1) == to_numpy(val_data[1]).argmax(1)).mean()
        self._untrain(self.X, self.y, self.num_iters, self.batch_size)
        
        # Return dummy loss, so we can propagate errors
        tmp = super(HyperLayer, self).__call__(self.lrs, self.mos)
        return tmp.sum()
        
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
        return torch.cat([lrs, mos], dim=1)
    
    @staticmethod
    def _backward_hook(self, grad_input, grad_output):
        """ pass gradients from hypersgd back to lrs/mos """
        return Variable(self.opt.d_lrs), Variable(self.opt.d_mos)

