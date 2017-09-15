#!/usr/bin/env python

"""
    simple_flat_hsgd.py
    
    HSGD, implemented by flattening the layers, w/ simplified implementation
    
    !! Needs to be tested
"""

from torch import autograd
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from exact_reps import *

class SimpleFlatHSGD(Optimizer):
    def __init__(self, params, lrs, mos, cuda=False):
        super(SimpleFlatHSGD, self).__init__()
        
        self.lrs = lrs
        self.mos = mos
        
        self.cuda    = cuda
        self.etensor = ETensorCUDA if cuda else ETensor
        
        self.params   = self.param_groups[0]['params']
        self._szs     = [np.prod(p.size()) for p in self.params]
        self._offsets = [0] + list(np.cumsum(self._szs))[:-1]
        self._numel   = sum([p.numel() for p in self.params])
        
        self.d_lrs = lrs.clone().zero_()
        self.d_mos = mos.clone().zero_()
        self.d_v   = torch.zeros(self._numel()).double()
        if self.cuda:
            self.d_v = self.d_v.cuda()

        # No sparse layers, yet
        for p in self.params:
            if p.data.is_sparse:
                raise NotImplemented

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
        
        assert offset == self._numel(), 'FlatHSGD._set_flat_params: offset != self._numel()'
    
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
        # !! This isn't going to support sparse layers
        return torch.cat([xx.contiguous().view(-1) for xx in x])
    
    def step(self, i):
        lr = self._fill_parser(self.lrs[i])
        mo = self._fill_parser(self.mos[i])
        
        flat_params = self._get_flat_params()
        flat_grad = self._get_flat_grads()
        
        if not self.eX:
            self.eX = self.etensor(flat_params.data.clone())
            self.eV = self.etensor(flat_grad.clone().zero_())
        
        _ = self.eV.mul(mo).sub(flat_grad)
        _ = self.eX.add(lr * self.eV.val)
        self._set_flat_params(self.eX.val)
    
    def init_backwards(self, lf):
        self.d_x = self._flatten(autograd.grad(lf(), self.params)).data
            
    def unstep(self, lf, i=0):
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

