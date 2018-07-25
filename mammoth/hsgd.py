#!/usr/bin/env python

"""
    hsgd.py
"""

import numpy as np

import torch
from torch import autograd
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

from .helpers import to_numpy
from .exact_reps import ETensor_torch as ETensor

class HSGD():
    def __init__(self, params, hparams, szs=None):
        """
            params:  parameters to optimize
            hparams: hyperparamters to optimize
        """
        
        assert 'lrs' in hparams, 'lrs' not in hparams
        assert 'mos' in hparams, 'mos' not in hparams
        
        self.params = list(params)
        
        # hparams
        self.lrs  = hparams.get('lrs', None)
        self.mos  = hparams.get('mos', None)
        self.meta = hparams.get('meta', None)
        
        # hparam derivatives
        self.d = {}
        for k,v in hparams.items():
            self.d[k] = v.data.clone().zero_()
        
        self.d_v = self._get_flat_params().data.clone().zero_()
        self.d_g = self._get_flat_params().data.clone().zero_()
        
        self._numel   = sum([p.numel() for p in self.params])
        self._shapes  = [p.shape for p in self.params]
        self._szs     = szs if szs is not None else [np.prod(p.size()) for p in self.params]
        self._offsets = [0] + list(np.cumsum(self._szs))[:-1]
        
        self.forward_ready = False
        self.backward_ready = False
        
        # No sparse layers, yet
        for p in self.params:
            assert not p.data.is_sparse
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def _flatten(self, x):
        return torch.cat([xx.contiguous().view(-1) for xx in x], dim=0)
    
    def _get_flat_params(self):
        return torch.cat([p.contiguous().view(-1) for p in self.params], dim=0)
    
    def _set_flat_params(self, val):
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.set_(val[offset:offset + numel].view_as(p.data))
            offset += numel
        
        assert offset == self._numel, 'FlatHSGD._set_flat_params: offset != self._numel()'
    
    def _fill_parser(self, vals):
        assert len(vals) == len(self._szs)
        return torch.cat([v.expand(int(s)) for v,s in zip(vals, self._szs)], dim=0).detach()
    
    def _get_flat_grads(self):
        return torch.cat([p.grad.data.view(-1) for p in self.params], dim=0)
    
    def step(self, sgd_iter):
        flat_params = self._get_flat_params()
        flat_grad   = self._get_flat_grads()
        
        if not self.forward_ready:
            self.eX = ETensor(flat_params.data.clone())
            self.eV = ETensor(flat_grad.clone().zero_())
            self.forward_ready = True
            
        lr = self._fill_parser(self.lrs[sgd_iter])
        mo = self._fill_parser(self.mos[sgd_iter])
        
        _ = self.eV.mul(mo).sub(flat_grad)
        _ = self.eX.add(lr * self.eV.val)
        _ = self._set_flat_params(self.eX.val)
    
    def init_backward(self, lf):
        assert self.forward_ready, 'cannot init_backward before calling HSGD.step'
        self.d_x = self._flatten(autograd.grad(lf(), self.params)).data
        self.g_data = self.d_x.clone()
        self.backward_ready = True
        
    def unstep(self, lf, sgd_iter):
        assert self.backward_ready, 'backward_ready = False'
        
        lr = self._fill_parser(self.lrs[sgd_iter])
        mo = self._fill_parser(self.mos[sgd_iter])
        
        # Update learning rate
        tmp = self.d_x * self.eV.val
        self.d['lrs'][sgd_iter] = torch.stack([tmp[offset:(offset+sz)].sum() for offset,sz in zip(self._offsets, self._szs)])
        
        # Reverse SGD exactly
        _ = self.eX.sub(lr * self.eV.val)
        _ = self._set_flat_params(self.eX.val)
        g = self._flatten(autograd.grad(lf(), self.params, create_graph=True))
        self.g_data = g.data
        _ = self.eV.add(g.data).unmul(mo)
        
        # Update mo
        self.d_v += self.d_x * lr
        tmp = self.d_v * self.eV.val
        self.d['mos'][sgd_iter] = torch.stack([tmp[offset:(offset+sz)].sum() for offset,sz in zip(self._offsets, self._szs)])
        
        # Update auxilliary parameters and (maybe) meta-parameters
        g = self._flatten(autograd.grad(lf(), self.params, create_graph=True))
        lf_hvp_x = (g * self.d_v).sum()
        self.d_x -= self._flatten(autograd.grad(lf_hvp_x, self.params)).data
        
        if self.meta is not None:
            g = self._flatten(autograd.grad(lf(), self.params, create_graph=True))
            lf_hvp_mts = (g * self.d_v).sum()
            self.d['meta'] -= self._flatten(autograd.grad(lf_hvp_mts, self.meta)).data
            
        self.d_v *= mo
    
    def get_init_params_grad(self):
        offset = 0
        out = []
        for shape in self._shapes:
            numel = np.prod(shape)
            tmp = self.d_x[offset:(offset+numel)].view(shape)
            out.append(tmp)
        
        return out