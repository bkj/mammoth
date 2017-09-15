#!/usr/bin/env python

"""
    flat_hsgd.py
    
    HSGD, implemented by flattening the layers
    
    Could very likely be sped up
"""

from torch.nn import Parameter
from torch import autograd
from torch.optim.optimizer import Optimizer

from exact_reps import *

class FlatHSGD(Optimizer):
    def __init__(self, params, lrs, momentums, cuda=False):
        super(FlatHSGD, self).__init__(params, {
            "lrs" : lrs,
            "momentums" : momentums,
        })
        
        self.d_lrs = lrs.clone().zero_()
        self.d_momentums = momentums.clone().zero_()
        
        self.cuda = cuda
        self.etensor = ETensorCUDA if cuda else ETensor
        
        self._params  = self.param_groups[0]['params']
        self._szs     = [np.prod(p.size()) for p in self._params]
        self._offsets = [0] + list(np.cumsum(self._szs))[:-1]
        self._numel_cache = None
        
        for p in self._params:
            if p.data.is_sparse:
                raise NotImplemented
        
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache
    
    def _get_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                # view = p.to_dense().view(-1)
                raise NotImplemented
            else:
                view = p.contiguous().view(-1)
            views.append(view)
        
        return torch.cat(views, 0)
    
    def _set_flat_params(self, val):
        offset = 0
        for p in self._params:
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
        for p in self._params:
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
        group = self.param_groups[0]
        lr = self._fill_parser(group['lrs'][i])
        momentum = self._fill_parser(group['momentums'][i])
        param_state = self.state['flat']
        
        flat_params = self._get_flat_params()
        flat_grad = self._get_flat_grads()
        
        if 'X' not in param_state:
            param_state['X'] = self.etensor(flat_params.data.clone())
        
        if 'V' not in param_state:
            param_state['V'] = self.etensor(flat_grad.clone().zero_())
        
        _ = param_state['V'].mul(momentum).sub(flat_grad)
        _ = param_state['X'].add(lr * param_state['V'].val)
        self._set_flat_params(param_state['X'].val)
    
    def init_backwards(self, lf):
        param_state = self.state['flat']
        param_state['d_x'] = self._flatten(autograd.grad(lf(), self._params)).data
        param_state['d_v'] = torch.zeros(self._numel()).double()
        if self.cuda:
            param_state['d_v'] = param_state['d_v'].cuda()
            
    def unstep(self, lf, i=0):
        group = self.param_groups[0]
        lr = self._fill_parser(group['lrs'][i])
        momentum = self._fill_parser(group['momentums'][i])
        # print 'momentum', to_numpy(momentum).sum()
        
        # Initialize parameters
        param_state = self.state['flat']
        # print 'd_x', to_numpy(param_state['d_x'])
        # print 'V', to_numpy(param_state['V'].val)
        
        # Update learning rate
        for j,(offset, sz) in enumerate(zip(self._offsets, self._szs)):
            self.d_lrs[i,j] = (param_state['d_x'][offset:(offset+sz)] * param_state['V'].val[offset:(offset+sz)]).sum()
        
        # print 'd_lrs', to_numpy(self.d_lrs[i])
        
        # Reverse SGD exactly
        _ = param_state['X'].sub(lr * param_state['V'].val)
        self._set_flat_params(param_state['X'].val)
        # print 'X', to_numpy(self._get_flat_params())
        
        g1 = self._flatten(autograd.grad(lf(), self._params)).data
        # print 'g1', to_numpy(g1)
        _ = param_state['V'].add(g1).div(momentum)
        # print 'V update', to_numpy(param_state['V'].val)
        
        # Update momentum
        param_state['d_v'] += param_state['d_x'] * lr
        for j,(offset, sz) in enumerate(zip(self._offsets, self._szs)):
            self.d_momentums[i,j] = (param_state['d_v'][offset:(offset+sz)] * param_state['V'].val[offset:(offset+sz)]).sum()
        
        # Update gradient
        d_vpar = Parameter(param_state['d_v'], requires_grad=True)
        g2 = (self._flatten(autograd.grad(lf(), self._params, create_graph=True)) * d_vpar).sum()
        param_state['d_x'] -= self._flatten(autograd.grad(g2, self._params)).data
        
        param_state['d_v'] = param_state['d_v'] * momentum
        # print

