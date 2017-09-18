#!/usr/bin/env pyton

"""
    exact_reps.py
    
    Wrappers for torch tensors that allow for exact (reversible) +,-,*,/
    
    !! Could probably be made faster, eg by reducing GPU<->CPU and numpy<->torch trips
    
    !! TODO -- add support for both doubles and floats
        - haven't tested w/ floats, but presumably works similarly
        - RADIX_SCALE maybe has to change though
"""

import sys
import torch
import numpy as np
from helpers import to_numpy

class ETensor_numpy(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        
        self.cuda = val.is_cuda
        self.intrep = self.float_to_intrep(val)
        self.size = val.size()
        
        self.aux = self._make_aux(self.size)
    
    def _make_aux(self, size):
        if len(size) > 1:
            return np.array([[0L] * size[1]] * size[0], dtype=object)
        else:
            return np.array([0L] * size[0], dtype=object)
    
    def _push_pop(self, r, n, d):
        assert torch.le(d, 2 ** 16).all(), 'ETensor._push_pop: M > 2 ** 16'
        
        d = to_numpy(d).astype(long)
        r = to_numpy(r).astype(long)
        n = to_numpy(n)
        
        self.aux *= d
        self.aux += r
        
        res = self.aux % n
        self.aux -= res
        self.aux /= n
        
        return torch.LongTensor(res)
    
    def add(self, a):
        self.intrep += self.float_to_intrep(a)
        return self
    
    def sub(self, a):
        self.add(-a)
        return self
    
    def rational_mul(self, n, d):
        r = self.intrep % d
        res = self._push_pop(r, n, d)
        if self.cuda:
            res = res.cuda()
        
        self.intrep -= self.intrep % d
        self.intrep /= d
        self.intrep *= n
        self.intrep += res
        return self
        
    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self
        
    def float_to_rational(self, a):
        assert torch.gt(a, 0.0).all()
        d = 2 ** 16 / torch.floor(a + 1).long()
        n = torch.floor(a * d.double() + 1).long()
        return n, d
        
    def float_to_intrep(self, x):
        intrep = (x * self.RADIX_SCALE).long()
        if self.cuda:
            intrep = intrep.cuda()
        return intrep
    
    @property
    def val(self):
        return self.intrep.double() / self.RADIX_SCALE


class ETensor_torch(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        
        print >> sys.stderr, 'ETensor -- vTorch'
        
        self.cuda = val.is_cuda
        self.intrep = self.float_to_intrep(val)
        self.size = val.size()
        
        self.aux = self._make_aux(self.size)
        self.aux_buffer = []
        self.aux_pushed_at = [0]
        
        self.counter = 0
    
    def _make_aux(self, size):
        if len(size) > 1:
            aux = np.array([[0L] * size[1]] * size[0], dtype=object)
        else:
            aux = np.array([0L] * size[0], dtype=object)
        
        return torch.LongTensor(aux).cuda()
    
    def _buffer(self, r, n, d):
        assert torch.le(d, 2 ** 16).all(), 'ETensor._push_pop: M > 2 ** 16'
        
        self.aux *= d
        self.aux += r
        
        resid = self.aux % n
        self.aux /= n
        
        return resid
    
    def add(self, a):
        self.intrep += self.float_to_intrep(a)
        return self
    
    def sub(self, a):
        self.add(-a)
        return self
    
    def rational_mul(self, n, d):
        r = self.intrep % d
        resid = self._buffer(r, n, d)
        if self.cuda:
            resid = resid.cuda()
        
        self.intrep -= self.intrep % d
        self.intrep /= d
        self.intrep *= n
        self.intrep += resid
        
    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        
        self.counter += 1
        # If true, then could overflow on next iteration
        if self.aux.max() > 2 ** (63 - 16):
            # print 'pushing aux @ %d' % self.counter
            self.aux_buffer.append(self.aux)
            self.aux = self._make_aux(self.size)
            self.aux_pushed_at.append(self.counter)
        
        return self
        
    def unmul(self, a):
        
        if self.counter == self.aux_pushed_at[-1]:
            assert (self.aux == 0).all()
            # print 'popping aux @ %d' % self.counter
            self.aux = self.aux_buffer.pop()
            _ = self.aux_pushed_at.pop()
        
        self.counter -= 1
        
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        
        return self
        
    def float_to_rational(self, a):
        assert torch.gt(a, 0.0).all()
        d = 2 ** 16 / torch.floor(a + 1).long()
        n = torch.floor(a * d.double() + 1).long()
        return n, d
        
    def float_to_intrep(self, x):
        intrep = (x * self.RADIX_SCALE).long()
        if self.cuda:
            intrep = intrep.cuda()
        return intrep
    
    @property
    def val(self):
        return self.intrep.double() / self.RADIX_SCALE

