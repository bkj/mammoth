#!/usr/bin/env pyton

"""
    exact_reps.py
    
    Wrappers for torch tensors that allow for exact (reversible) +,-,*,/
    
    !! Could probably be made faster, eg by reducing GPU<->CPU and numpy<->torch trips
"""

import sys
import torch
import numpy as np
from .helpers import to_numpy

float_cast = lambda x: x.float()
# double_cast = lambda x: x.double()

class _ETensor_torch(object):
    RADIX_SCALE = int(2 ** 52)
    def __init__(self, val):
        self.cuda = val.is_cuda
        self._cast = float_cast
        
        self.intrep = self.float_to_intrep(val)
        self.size = val.size()
        
        self.aux = self.val.clone().zero_().long() # !! Fastest way?
        self.aux_buffer = []
        self.aux_pushed_at = [0]
        
        self.counter = 0
    
    def _buffer(self, r, n, d):
        assert torch.le(d, 2 ** 16).all(), 'ETensor_torch._buffer: M > 2 ** 16'
        
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
    
    def float_to_rational(self, a):
        assert torch.gt(a, 0.0).all()
        d = (2 ** 16 / torch.floor(a + 1)).long()
        n = torch.floor(a * self._cast(d) + 1).long()
        return n, d
    
    def float_to_intrep(self, x):
        intrep = (x * self.RADIX_SCALE).long()
        if self.cuda:
            intrep = intrep.cuda()
        return intrep
    
    @property
    def val(self):
        return self._cast(self.intrep) / self.RADIX_SCALE


class ETensor_torch(_ETensor_torch):
    """
        Original implementation -- not optimally space efficient
    """
    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        
        self.counter += 1
        # If true, then could overflow on next iteration
        if self.aux.max() > 2 ** (63 - 16):
            self.aux_buffer.append(self.aux)
            self.aux = self.aux.clone().zero_() # !! Fastest way?
            self.aux_pushed_at.append(self.counter)
        
        return self
    
    def unmul(self, a):
        
        if self.counter == self.aux_pushed_at[-1]:
            assert (self.aux == 0).all()
            self.aux = self.aux_buffer.pop()
            _ = self.aux_pushed_at.pop()
        
        self.counter -= 1
        
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        
        return self


# class ETensor_torch(_ETensor_torch):
#     """ 
#         Alternate implementation that's more space efficient.
#         Little more complicated, though so let's punt for now.
#     """
#     def mul(self, a):
#         n, d = self.float_to_rational(a)
#         self.rational_mul(n, d)
        
#         self.counter += 1
#         # If true, then could overflow on next iteration
#         if self.aux.max() > 2 ** (63 - 16):
#             mask = self.aux > 2 ** (63 - 16)
#             self.aux_buffer.append((mask, self.aux.masked_select(mask)))
#             self.aux.masked_fill_(mask, 0)
#             self.aux_pushed_at.append(self.counter)
        
#         return self
    
#     def unmul(self, a):
#         if self.counter == self.aux_pushed_at[-1]:
#             mask, old_entries = self.aux_buffer.pop()
#             self.aux.masked_scatter_(mask, old_entries)
#             _ = self.aux_pushed_at.pop()
        
#         self.counter -= 1
        
#         n, d = self.float_to_rational(a)
#         self.rational_mul(d, n)
        
#         return self
