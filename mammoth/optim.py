#!/usr/bin/env python

"""
    optim.py
"""

import math
import torch
import numpy as np

class LambdaAdam(torch.optim.Optimizer):
    """
        ADAM optimizer that mimics hypergrads
        - Difference is addition of `lam` parameter.  I noticed that my hypergrad test was converging
        to eps < 1e-10.  Setting lam to some small number (1e-1, 1e-2, etc) lets the torch version
        convert to eps < 1e-8.
        
        !! This is not efficient, due to cloning, etc.  Will need to reimplement more efficiently
        for larger models.  Then again, for larger models, this may not matter.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=10**-4, lam=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, lam=lam)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                if len(state) == 0:
                    state['step']       = 0
                    state['exp_avg']    = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                        
                m, v = state['exp_avg'].clone(), state['exp_avg_sq'].clone()
                
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # --
                
                b1t  = beta1 * (group['lam'] ** (state['step'] - 1))
                m    = (m * b1t) + ((1 - b1t) * grad)
                v    = (1 - beta2) * (grad ** 2) + beta2 * v
                mhat = m / (1 - beta1 ** state['step'])
                vhat = v / (1 - beta2 ** state['step'])
                p.data -= group['lr'] * mhat / (torch.sqrt(vhat) + group['eps'])
                
                # --
                # default torch implementation
                
                # m     = (m * beta1) + ((1 - beta1) * grad)
                # v     = (1 - beta2) * (grad ** 2) + beta2 * v
                # denom = torch.sqrt(v) + group['eps']
                
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # p.data.addcdiv_(-step_size, m, denom)
                
                # --
                
                state['exp_avg'] = m.clone()
                state['exp_avg_sq'] = v.clone()
                
        return loss