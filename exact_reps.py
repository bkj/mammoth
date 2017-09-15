
import torch
import numpy as np

class ETensor(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        assert isinstance(val, torch.DoubleTensor)
        self.intrep = self.float_to_intrep(val)
        self.aux = EBitStore(val.size())
        
    def add(self, a):
        assert isinstance(a, torch.DoubleTensor)
        self.intrep += self.float_to_intrep(a)
        return self
        
    def sub(self, a):
        assert isinstance(a, torch.DoubleTensor)
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
        assert isinstance(a, torch.DoubleTensor)
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        assert isinstance(a, torch.DoubleTensor)
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self
        
    def float_to_rational(self, a):
        assert isinstance(a, torch.DoubleTensor)
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
        assert torch.le(M, 2 ** 16).all()
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
        assert isinstance(val, torch.cuda.DoubleTensor)
        self.intrep = self.float_to_intrep(val)
        self.aux = EBitStoreCUDA(val.size())
        
    def add(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor)
        self.intrep += self.float_to_intrep(a)
        return self
        
    def sub(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor)
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
        assert isinstance(a, torch.cuda.DoubleTensor)
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self
        
    def div(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor)
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self
        
    def float_to_rational(self, a):
        assert isinstance(a, torch.cuda.DoubleTensor)
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
        assert torch.le(M, 2 ** 16).all()
        self.store *= to_numpy(M).astype(long)
        self.store += to_numpy(N).astype(long)
        
    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % to_numpy(M)
        self.store /= to_numpy(M)
        return torch.LongTensor(N).cuda()




# --
# For autograd

class ExactRep(object):
    RADIX_SCALE = long(2 ** 52)
    def __init__(self, val, from_intrep=False):
        if from_intrep:
            self.intrep = val
        else:
            self.intrep = self.float_to_intrep(val)
            
        self.aux = BitStore(len(val))
        
    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += self.float_to_intrep(A)
        return self
        
    def sub(self, A):
        self.add(-A)
        return self
        
    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d) # Store remainder bits externally
        self.intrep /= d                  # Divide by denominator
        self.intrep *= n                  # Multiply by numerator
        nn = self.aux.pop(n).astype(long)
        self.intrep += nn
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
        assert np.all(a > 0.0)
        d = 2**16 / np.fix(a+1).astype(long) # Uglier than it used to be: np.int(a + 1)
        n = np.fix(a * d + 1).astype(long)
        return  n, d
        
    def float_to_intrep(self, x):
        return (x * self.RADIX_SCALE).astype(np.int64)
        
    @property
    def val(self):
        return self.intrep.astype(np.float64) / self.RADIX_SCALE


class BitStore(object):
    """Efficiently stores information with non-integer number of bits (up to 16)."""
    def __init__(self, length):
        # Use an array of Python 'long' ints which conveniently grow
        # as large as necessary. It's about 50X slower though...
        self.store = np.array([0L] * length, dtype=object)
        
    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        assert np.all(M <= 2 ** 16)
        self.store *= M
        self.store += N
        
    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        self.store /= M
        print self.store
        return N


