#!/usr/bin/env python

"""
    run-flat-hsgd.py
"""

import sys
sys.path.append('/home/bjohnson/software/autograd/')
sys.path.append('/home/bjohnson/software/hypergrad')

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from hypergrad.data import load_data_dicts
from hypergrad.util import RandomState

from rsub import *
from matplotlib import pyplot as plt

from helpers import to_numpy
# from simple_flat_hsgd import FlatHSGD

# --
# IO

batch_size  = 200
num_iters   = 20
N_classes   = 10
N_train     = 10000
N_valid     = 10000
N_tests     = 10000

train_data, valid_data, _ = load_data_dicts(N_train, N_valid, N_tests)

X_train = Variable(torch.DoubleTensor(train_data['X'])).cuda()
y_train = Variable(torch.LongTensor(train_data['T'].argmax(axis=1))).cuda()

X_val = Variable(torch.DoubleTensor(valid_data['X'])).cuda()
y_val = valid_data['T'].argmax(axis=1)

# --
# Helpers

logit = lambda x: 1 / (1 + (-x).exp())
d_logit = lambda x: x.exp() / ((1 + x.exp()) ** 2)
d_exp = lambda x: x.exp()

def set_net_weights(net, val):
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        if len(p.size()) == 1:
            p.data.set_(val[offset:offset + numel].view_as(p.data))
        else:
            p.data.set_(val[offset:offset + numel].view_as(p.data.t()).t()) # !! autograd ordering
        offset += numel
    print offset

def make_net(weight_scale=np.exp(-3), layers=[50, 50, 50]):
    
    net = nn.Sequential(
        nn.Linear(784, layers[0]),
        nn.Tanh(),
        nn.Linear(layers[0], layers[1]),
        nn.Tanh(),
        nn.Linear(layers[1], layers[2]),
        nn.Tanh(),
        nn.Linear(layers[2], 10),
        nn.LogSoftmax(),
    )
    
    for child in net.children():
        if isinstance(child, nn.Linear):
            _ = child.weight.data.normal_(0, weight_scale)
    
    net = net.double()
    return net


def deterministic_batch(X, y, sgd_iter, meta_iter, seed=0, batch_size=batch_size):
    rs = RandomState((seed, meta_iter, sgd_iter))
    idxs = rs.randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    X, y = X[idxs], y[idxs]
    return X, y


def train(net, opt, num_iters, meta_iter, seed=0):
    train_hist, val_hist = [], []
    for i in tqdm(range(num_iters)):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        
        opt.step(i) if isinstance(opt, FlatHSGD) else opt.step()
        
        # train_acc = (to_numpy(net(X_train)).argmax(1) == to_numpy(y_train)).mean()
        # train_hist.append(train_acc)
        
        # val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
        # val_hist.append(val_acc)
    
    return opt, val_hist, train_hist


def untrain(net, opt, num_iters, meta_iter, seed=0):
    for i in tqdm(range(num_iters)[::-1]):
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.unstep(lf, i)
    
    return opt

# >>


#!/usr/bin/env pyton

"""
    exact_reps.py
    
    Wrappers for torch tensors that allow for exact (reversible) +,-,*,/
    
    !! Could probably be made faster, eg by reducing GPU<->CPU and numpy<->torch trips
    
    !! TODO -- add support for both doubles and floats
        - haven't tested w/ floats, but presumably works similarly
        - RADIX_SCALE maybe has to change though
"""

import torch
import numpy as np
from helpers import to_numpy

class ETensor(object):
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

# <<


# --
# Run

meta_iter = 0
num_iters = 100
lrs = torch.DoubleTensor(np.full((num_iters, 8), -1.0)).cuda()
mos = torch.DoubleTensor(np.full((num_iters, 8), 0.0)).cuda()

net = make_net(
    weight_scale=np.exp(-3),
    layers=[100] * 3
).cuda()

opt = torch.optim.SGD(net.parameters(),
    lr=0.1,
    momentum=0.9
)
t = time()
opt, val_hist, train_hist = train(net, opt, num_iters=num_iters, meta_iter=meta_iter, seed=0)
print 'train=%f' % (time() - t)


opt = FlatHSGD(net.parameters(),
    lrs=lrs.exp(),
    mos=logit(mos)
)
t = time()
opt, val_hist, train_hist = train(net, opt, num_iters=num_iters, meta_iter=meta_iter, seed=0)
print 'train=%f' % (time() - t)

from numba import jit

@jit
def f(x):
    for i in range(len(x)):
        x[i] += 2 ** 100
    return x


x = list(np.random.choice(100, 100000))
t = time()
z = f(x)
time() - t

z[0]


# trained_weights = to_numpy(opt._get_flat_params())

# # Init untrain
# def lf_all():
#     return F.cross_entropy(net(X_train), y_train)

# opt.init_backward(lf_all)

# # Untrain
# t = time()
# opt = untrain(net, opt, num_iters, meta_iter)
# untrained_weights = to_numpy(opt._get_flat_params())
# assert np.all(orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
# print 'untrain=%f' % (time() - t)