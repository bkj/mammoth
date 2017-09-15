#!/usr/bin/env pyton

"""
    exact_reps.py
    
    Wrappers for torch tensors that allow for exact (reversible) +,-,*,/
    
    !! Could probably be made faster, eg by reducing GPU<->CPU and numpy<->torch trips
"""

import torch
import numpy as np
from helpers import to_numpy

class ETensor(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        assert isinstance(val, torch.DoubleTensor), 'ETensor.__init__: type(a) != torch.DoubleTensor'
        self.intrep = self.float_to_intrep(val)
        self.aux = EBitStore(val.size())
        
    def add(self, a):
        assert isinstance(a, torch.DoubleTensor), 'ETensor.add: type(a) != torch.DoubleTensor'
        self.intrep += self.float_to_intrep(a)
        return self
        
    def sub(self, a):
        assert isinstance(a, torch.DoubleTensor), 'ETensor.sub: type(a) != torch.DoubleTensor'
        self.add(-a)
        return self
        
    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d)
        self.intrep -= self.intrep % d
        self.intrep /= d
        self.intrep *= n
        self.intrep += self.aux.pop(n)
        return self
        
    def mul(self, a):
        assert isinstance(a, torch.DoubleTensor), 'ETensor.mul: type(a) != torch.DoubleTensor'
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        assert isinstance(a, torch.DoubleTensor), 'ETensor.div: type(a) != torch.DoubleTensor'
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self
        
    def float_to_rational(self, a):
        assert isinstance(a, torch.DoubleTensor), 'ETensor.float_to_rational: type(a) != torch.DoubleTensor'
        assert torch.gt(a, 0.0).all()
        d = 2 ** 16 / torch.floor(a + 1).long()
        n = torch.floor(a * d.double() + 1).long()
        return n, d
        
    def float_to_intrep(self, x):
        return (x * self.RADIX_SCALE).long()
    
    @property
    def val(self):
        return self.intrep.double() / self.RADIX_SCALE
    
    @property
    def size(self):
        return self.val.size()


class EBitStore(object):
    """
        Efficiently stores information with non-integer number of bits (up to 16).
    """
    def __init__(self, size):
        if len(size) > 1:
            self.store = np.array([[0L] * size[1]] * size[0], dtype=object)
        else:
            self.store = np.array([0L] * size[0], dtype=object)
        
    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert torch.le(M, 2 ** 16).all(), 'EBitStore.push: M > 2 ** 16'
        self.store *= to_numpy(M).astype(long)
        self.store += to_numpy(N).astype(long)
        
    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % to_numpy(M)
        self.store /= to_numpy(M)
        return torch.LongTensor(N)

# --
# Cuda
class ETensorCUDA(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        assert isinstance(val, torch.cuda.DoubleTensor), 'ETensorCUDA.__init__: type(a) != torch.cuda.DoubleTensor'
        self.intrep = self.float_to_intrep(val)
        self.aux = EBitStoreCUDA(val.size())
        
    def add(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor), 'ETensorCUDA.add: type(a) != torch.cuda.DoubleTensor'
        self.intrep += self.float_to_intrep(a)
        return self
        
    def sub(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor), 'ETensorCUDA.sub: type(a) != torch.cuda.DoubleTensor'
        self.add(-a)
        return self
        
    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d)
        self.intrep -= self.intrep % d
        self.intrep /= d
        self.intrep *= n
        self.intrep += self.aux.pop(n)
        return self
        
    def mul(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor), 'ETensorCUDA.mul: type(a) != torch.cuda.DoubleTensor'
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor), 'ETensorCUDA.div: type(a) != torch.cuda.DoubleTensor'
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self
        
    def float_to_rational(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor), 'ETensorCUDA.float_to_rational: type(a) != torch.cuda.DoubleTensor'
        assert torch.gt(a, 0.0).all()
        d = 2 ** 16 / torch.floor(a + 1).long()
        n = torch.floor(a * d.double() + 1).long()
        return n, d
        
    def float_to_intrep(self, x):
        return (x * self.RADIX_SCALE).long().cuda()
    
    @property
    def val(self):
        return self.intrep.double().cuda() / self.RADIX_SCALE
    
    @property
    def size(self):
        return self.val.size()


class EBitStoreCUDA(object):
    """
        Efficiently stores information with non-integer number of bits (up to 16).
    """
    def __init__(self, size):
        if len(size) > 1:
            self.store = np.array([[0L] * size[1]] * size[0], dtype=object)
        else:
            self.store = np.array([0L] * size[0], dtype=object)
        
    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert torch.le(M, 2 ** 16).all(), 'EBitStoreCUDA.push: M > 2 ** 16'
        self.store *= to_numpy(M).astype(long)
        self.store += to_numpy(N).astype(long)
        
    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % to_numpy(M)
        self.store /= to_numpy(M)
        return torch.LongTensor(N).cuda()
