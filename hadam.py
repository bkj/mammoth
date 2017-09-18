#!/usr/bin/env python

"""
    hadam.py
    
    ADAM-based hyperoptimizer
    
    !! Not actually different than regular ADAM, other than
        a) how it handles shapes of parameteres
        b) gradients get passed in manually, rather than w/ autograd
"""

import torch
import numpy as np

class HADAM(object):
    def __init__(self, params, step_size=0.1, b1=0.1, b2=0.01, eps=10**-4, lam=10**-4):
        
        self._flat_params = torch.cat([p.contiguous().view(-1) for p in params])
        
        self._sizes = [p.size() for p in params]
        self._numel = sum([p.numel() for p in params])
        
        self.step_size = step_size
        self.b1        = b1
        self.b2        = b2
        self.eps       = eps
        self.lam       = lam
        self.m         = self._flat_params.clone().zero_()
        self.v         = self._flat_params.clone().zero_()
        self.iter      = 0
    
    @property
    def params(self):
        offset = 0
        for size in self._sizes:
            numel = np.prod(size)
            yield self._flat_params[offset:offset + numel].view(size)
            offset += numel
        
        assert offset == self._numel, 'HyperADAM: offset != self._numel'
        
    def step_w_grads(self, grads):
        g = torch.cat([grad.contiguous().view(-1) for grad in grads])
        
        b1t = 1 - (1 - self.b1) * (self.lam ** self.iter)
        self.m = b1t * g + (1 - b1t) * self.m
        self.v = self.b2 * (g ** 2) + (1 - self.b2) * self.v
        mhat = self.m / (1 - (1 - self.b1) ** (self.iter + 1))
        vhat = self.v / (1 - (1 - self.b2) ** (self.iter + 1))
        
        self.iter += 1
        self._flat_params -= self.step_size * mhat / (vhat.sqrt() + self.eps)
        return self.params