#!/usr/bin/env python

"""
    hsgd.py
    
    HSGD, on a per-layer basis
    
    !! Prefer flat_hsgd.py if possible
    !! This is buggy, I think
    !! No CUDA support
"""

import torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer

class HSGD(Optimizer):
    def __init__(self, params, lrs, momentums, num_iters, cuda=False):
        super(HSGD, self).__init__(params, {
            "lrs" : lrs,
            "momentums" : momentums,
        })
        self.d_lrs       = torch.zeros(num_iters).double()
        self.d_momentums = torch.zeros(num_iters).double()
        
        self.cuda = cuda
        if self.cuda:
            self.d_lrs = self.d_lrs.cuda()
            self.d_momentums = self.d_momentums.cuda()
        
    def step(self, i):
        for group in self.param_groups:
            momentum = torch.DoubleTensor([group['momentums'][i]])
            lr = torch.DoubleTensor([group['lrs'][i]])
            
            if self.cuda:
                momentum = momentum.cuda()
                lr = lr.cuda()
                
            for param in group['params']:
                if param.grad is None:
                    continue
                
                g = param.grad.data
                
                param_state = self.state[param]
                
                if 'X' not in param_state:
                    if self.cuda:
                        param_state['X'] = ETensorCUDA(param.data.clone())
                    else:
                        param_state['X'] = ETensor(param.data.clone())
                    
                if 'V' not in param_state:
                    if self.cuda:
                        param_state['V'] = ETensorCUDA(g.clone().zero_())
                    else:
                        param_state['V'] = ETensor(g.clone().zero_())
                
                _ = param_state['V'].mul(momentum).sub(g)
                _ = param_state['X'].add(lr * param_state['V'].val)
                param.data.set_(param_state['X'].val)
        
    def unstep(self, lf, i=0):
        for group in self.param_groups:
            
            momentum = torch.DoubleTensor([group['momentums'][i]])
            lr = torch.DoubleTensor([group['lrs'][i]])
            if cuda:
                momentum = momentum.cuda()
                lr = lr.cuda()
            
            # Update parameters in all layers
            for param in group['params']:
                param_state = self.state[param]
                
                if 'd_x' not in param_state:
                    param_state['d_x'] = autograd.grad(lf(), param)[0].data
                
                if 'grad_proj_x' not in param_state:
                    param_state['grad_proj_x'] = lambda x, d: (autograd.grad(lf(), x, create_graph=True)[0] * d).sum()
                
                if 'd_v' not in param_state:
                    param_state['d_v'] = torch.zeros(param.size()).double()
                
                self.d_lrs[i] += (param_state['d_x'] * param_state['V'].val).sum()
                
                _ = param_state['X'].sub(lr * param_state['V'].val)
                param.data.set_(param_state['X'].val)
            
            # Update velocities in all layers
            for param in group['params']:
                param_state = self.state[param]
                g = autograd.grad(lf(), param)[0].data
                _ = param_state['V'].add(g).div(momentum)
                
                param_state['d_v'] += param_state['d_x'] * lr
                
                self.d_momentums[i] += (param_state['d_v'] * param_state['V'].val).sum()
                
                d_vpar = Parameter(param_state['d_v'], requires_grad=True)
                param_state['d_x'] -= autograd.grad(param_state['grad_proj_x'](param, d_vpar), param)[0].data
                
                param_state['d_v'] = param_state['d_v'] * momentum