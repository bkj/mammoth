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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class HSGD():
    def __init__(self, params, hparams, szs=None, learn_lrs=True, learn_mos=True, learn_meta=True):
        """
            params:  parameters to optimize
            hparams: hyperparamters to optimize
        """
        
        assert 'lrs' in hparams, "'lrs' not in hparams"
        assert 'mos' in hparams, "'mos' not in hparams"
        if learn_meta:
            assert 'meta' in hparams, "'meta' not in hparams"
        
        self.params     = list(params)
        
        self.learn_lrs  = learn_lrs
        self.learn_mos  = learn_mos
        self.learn_meta = learn_meta
        
        self.lrs  = hparams['lrs']
        self.mos  = hparams['mos']
        self.meta = hparams.get('meta', None)
        
        self.d_lrs  = self.lrs.data.clone().zero_() if self.learn_lrs else None
        self.d_mos  = self.mos.data.clone().zero_() if self.learn_mos else None
        self.d_meta = self.meta.data.clone().zero_() if self.learn_meta else None
        
        self.d_v = self._get_flat_params().data.clone().zero_()
        self.d_g = self._get_flat_params().data.clone().zero_()
        
        self._numel   = sum([p.numel() for p in self.params])
        self._shapes  = [p.shape for p in self.params]
        self._szs     = szs if szs is not None else [np.prod(p.size()) for p in self.params]
        self._offsets = [0] + list(np.cumsum(self._szs))[:-1]
        
        self.forward_ready  = False
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
        
        _ = self.eV.mul(mo).sub((1.0 - mo) * flat_grad)
        _ = self.eX.add(lr * self.eV.val)
        _ = self._set_flat_params(self.eX.val)
    
    def init_backward(self, lf_init=None, grad=False, n=0):
        assert self.forward_ready, 'cannot init_backward before calling HSGD.step'
        if lf_init is not None:
            self.d_x = self._flatten(autograd.grad(lf_init(), self.params)).data
            if self.learn_meta:
                self.d_meta = self._flatten(autograd.grad(lf_init(), self.meta)).data
        else:
            raise Exception
            # assert n > 0
            # self.d_x = self._flatten([p.grad for p in self.params]).data / n
            # if self.learn_meta:
            #     print(self.meta)
            #     self.d_meta = self._flatten([p.grad for p in self.meta]).data / n
        
        self.backward_ready = True
        
    def unstep(self, lf, sgd_iter):
        assert self.backward_ready, 'backward_ready = False'
        
        lr = self._fill_parser(self.lrs[sgd_iter])
        mo = self._fill_parser(self.mos[sgd_iter])
        
        # Update learning rate
        if self.learn_lrs:
            tmp = self.d_x * self.eV.val
            self.d_lrs[sgd_iter] = torch.stack([tmp[offset:(offset+sz)].sum() for offset,sz in zip(self._offsets, self._szs)])
        
        # Reverse SGD exactly
        _ = self.eX.sub(lr * self.eV.val)
        _ = self._set_flat_params(self.eX.val)
        g = self._flatten(autograd.grad(lf(), self.params, create_graph=True))
        _ = self.eV.add((1 - mo) * g.data).unmul(mo)
        
        # self.d_v += self.d_x * lr
        
        # # Update mo
        # if self.learn_mos:
        #     tmp = self.d_v * (self.eV.val + g.data)
        #     self.d_mos[sgd_iter] = torch.stack([tmp[offset:(offset+sz)].sum() for offset,sz in zip(self._offsets, self._szs)])
        
        # # Weight gradient
        # lf_hvp = (g * ((1 - mo) * self.d_v)).sum()
        # self.d_x -= self._flatten(autograd.grad(lf_hvp, self.params, retain_graph=True)).data
        
        # # Meta gradient
        # if self.learn_meta:
        #     d_meta_update = self._flatten(autograd.grad(lf_hvp, self.meta)).data
        #     self.d_meta -= d_meta_update
        
        # self.d_v *= mo
    
    def get_init_params_grad(self):
        offset = 0
        out = []
        for shape in self._shapes:
            numel = np.prod(shape)
            tmp = self.d_x[offset:(offset+numel)].view(shape)
            out.append(tmp)
        
        return out