#!/usr/bin/env python

"""
    simple_flat_hsgd.py
    
    HSGD, implemented by flattening the layers, w/ simplified implementation
    
    !! Needs to be tested
"""

import numpy as np

import torch
from torch import autograd
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from exact_reps import *

class FlatHSGD():
    def __init__(self, params, lrs, mos):
        self.params = list(params)
        self.cuda = self.params[0].is_cuda
        
        self.lrs = lrs if not self.cuda else lrs.cuda()
        self.mos = mos if not self.cuda else mos.cuda()
        
        self._szs     = [np.prod(p.size()) for p in self.params]
        self._offsets = [0] + list(np.cumsum(self._szs))[:-1]
        self._numel   = sum([p.numel() for p in self.params])
        
        self.d_lrs = lrs.clone().zero_()
        self.d_mos = mos.clone().zero_()
        self.d_v   = self._get_flat_params().data.clone().zero_()
        
        self.forward_ready = False
        self.backward_ready = False
        
        # No sparse layers, yet
        for p in self.params:
            if p.data.is_sparse:
                raise NotImplemented
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())
    
    def _get_flat_params(self):
        views = []
        for p in self.params:
            if p.data.is_sparse:
                # view = p.to_dense().view(-1)
                raise NotImplemented
            else:
                view = p.contiguous().view(-1)
            views.append(view)
        
        return torch.cat(views, 0)
    
    def _set_flat_params(self, val):
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.set_(val[offset:offset + numel].view_as(p.data))
            offset += numel
        
        assert offset == self._numel, 'FlatHSGD._set_flat_params: offset != self._numel()'
    
    def _fill_parser(self, vals):
        assert len(vals) == len(self._szs), 'FlatHSGD._fill_parser: len(vals) != len(self._szs)'
        views = []
        for i, s in enumerate(self._szs):
            view = vals.index(i).repeat(s)
            views.append(view)
        
        return torch.cat(views, 0)
    
    def _get_flat_grads(self):
        views = []
        for p in self.params:
            if p.grad is None:
                # view = p.data.new(p.data.numel()).zero_()
                raise NotImplemented
            elif p.grad.data.is_sparse:
                # view = p.grad.data.to_dense().view(-1)
                raise NotImplemented
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        
        return torch.cat(views, 0)
    
    def _flatten(self, x):
        # !! This doesn't support sparse layers
        return torch.cat([xx.contiguous().view(-1) for xx in x])
    
    def step(self, i):
        lr = self._fill_parser(self.lrs[i])
        mo = self._fill_parser(self.mos[i])
        
        flat_params = self._get_flat_params()
        flat_grad = self._get_flat_grads()
        
        if not self.forward_ready:
            self.eX = ETensor(flat_params.data.clone())
            self.eV = ETensor(flat_grad.clone().zero_())
            self.forward_ready = True
        
        _ = self.eV.mul(mo).sub(flat_grad)
        _ = self.eX.add(lr * self.eV.val)
        self._set_flat_params(self.eX.val)
    
    def init_backward(self, lf):
        self.d_x = self._flatten(autograd.grad(lf(), self.params)).data
        self.backward_ready = True
            
    def unstep(self, lf, i=0):
        assert self.backward_ready, 'backward_ready = False'
        
        lr = self._fill_parser(self.lrs[i])
        mo = self._fill_parser(self.mos[i])
        
        # Update learning rate
        for j,(offset, sz) in enumerate(zip(self._offsets, self._szs)):
            self.d_lrs[i,j] = (self.d_x[offset:(offset+sz)] * self.eV.val[offset:(offset+sz)]).sum()
        
        # Reverse SGD exactly
        _ = self.eX.sub(lr * self.eV.val)
        self._set_flat_params(self.eX.val)
        g1 = self._flatten(autograd.grad(lf(), self.params)).data
        _ = self.eV.add(g1).div(mo)
        
        # Update mo
        self.d_v += self.d_x * lr
        for j,(offset, sz) in enumerate(zip(self._offsets, self._szs)):
            self.d_mos[i,j] = (self.d_v[offset:(offset+sz)] * self.eV.val[offset:(offset+sz)]).sum()
        
        # Update auxilliary parameters
        d_vpar   = Parameter(self.d_v, requires_grad=True)
        lf_hvp_x = (self._flatten(autograd.grad(lf(), self.params, create_graph=True)) * d_vpar).sum()
        self.d_x -= self._flatten(autograd.grad(lf_hvp_x, self.params)).data
        self.d_v = self.d_v * mo

