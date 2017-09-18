#!/usr/bin/env python

"""
    run-2.py
    
    !! Need to clean this up so that arguments are passed around sanely
    !! Need to implement own version of `load_data_dicts` and `RandomState`
    !! Need to implement example where meta-parameters get optimized.
        - Could do this by implementing "scaling layer" in nn.Sequential
"""

import sys
sys.path.append('/home/bjohnson/software/hypergrad')

import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from hypergrad.util import RandomState

from rsub import *
from matplotlib import pyplot as plt

from helpers import to_numpy
from hsgd import HSGD
from hadam import HADAM

np.random.seed(123)
_ = torch.manual_seed(456)
_ = torch.cuda.manual_seed(789)

# !! Appears to be necessary 
torch.backends.cudnn.enabled = False

# --
# IO

batch_size  = 200
num_iters   = 100
train_size  = 40000

from keras.datasets import mnist
(X_train, y_train), (X_val, y_val) = mnist.load_data()

X_train = Variable(torch.DoubleTensor(X_train.astype('float'))).cuda()
y_train = Variable(torch.LongTensor(y_train.astype('int'))).cuda()

X_val = Variable(torch.DoubleTensor(X_val.astype('float'))).cuda()
y_val = y_val.astype('int')

X_train = X_train.view(X_train.size(0), 1, 28, 28)
X_val = X_val.view(X_val.size(0), 1, 28, 28)

X_train = X_train.expand(X_train.size(0), 3, 28, 28)
X_val = X_val.expand(X_val.size(0), 3, 28, 28)

X_train, X_val = X_train / 255, X_val / 255

X_train, y_train = X_train[:train_size], y_train[:train_size]

# --
# Helpers

logit = lambda x: 1 / (1 + (-x).exp())
d_logit = lambda x: x.exp() / ((1 + x.exp()) ** 2) # derivative of logit
d_exp = lambda x: x.exp() # derivative of exponent

def deterministic_batch(X, y, sgd_iter, meta_iter, seed=0, batch_size=batch_size):
    rs = RandomState((seed, meta_iter, sgd_iter))
    idxs = rs.randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    X, y = X[idxs], y[idxs]
    return X, y


def train(net, opt, num_iters, meta_iter, seed=0):
    hist = defaultdict(list)
    gen = tqdm(range(num_iters))
    for i in gen:
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        opt.zero_grad()
        scores = net(X)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        
        # batch_acc = (to_numpy(scores.max(1)[1]) == to_numpy(y)).mean()
        # gen.set_postfix({'batch_acc' : batch_acc, 'mode' : 'forward', 'meta_iter' : meta_iter})
        
        opt.step(i) if isinstance(opt, HSGD) else opt.step()
        
        # train_acc = (to_numpy(net(X_train)).argmax(1) == to_numpy(y_train)).mean()
        # val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
        
        # hist['train'].append(train_acc)
        # hist['val'].append(val_acc)
    
    val_acc = (to_numpy(net(X_val)).argmax(1) == y_val).mean()
    hist['val'].append(val_acc)
    
    return opt, hist


def untrain(net, opt, num_iters, meta_iter, seed=0):
    gen = tqdm(range(num_iters)[::-1])
    for i in gen:
        X, y = deterministic_batch(X_train, y_train, sgd_iter=i, meta_iter=meta_iter, seed=0)
        
        def lf():
            return F.cross_entropy(net(X), y)
        
        opt.unstep(lf, i)
        # gen.set_postfix({'mode' : 'backwards', 'meta_iter' : meta_iter})
    
    return opt


def do_meta_iter(meta_iter, net, lrs, mos):
    opt = HSGD(params=net.parameters(), lrs=lrs, mos=mos)
    
    orig_weights = to_numpy(opt._get_flat_params())
    
    # Train
    opt, hist = train(net, opt, num_iters=num_iters, meta_iter=meta_iter, seed=0)
    # print {"train_acc" : hist['train'][-1], "val_acc" : hist['val'][-1]}
    print 'val_acc=%f' % hist['val'][-1]
    
    # Init untrain
    def lf_all():
        return F.cross_entropy(net(X_train), y_train)
        
    opt.init_backward(lf_all)
    
    # Untrain
    opt = untrain(net, opt, num_iters, meta_iter)
    
    # Make sure SGD was reversed exactly
    untrained_weights = to_numpy(opt._get_flat_params())
    assert np.all(orig_weights == untrained_weights), 'meta_iter: orig_weights != untrained_weights'
    
    return opt, hist


# --
# Define network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 512)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# --
# Run

meta_iters = 50
step_size = 0.01

# Initial learning rates -- parameterized as log(lr)
lrs = torch.DoubleTensor(np.full((num_iters, 8), -2.3)).cuda()

# Initial momentums -- parameterized as inverse_logit(mo)
mos = torch.DoubleTensor(np.full((num_iters, 8), 2.5)).cuda()

# Hyper-ADAM optimizer
hyperopt = HADAM([lrs, mos], step_size=step_size)

# Run hypertraining
all_hist = defaultdict(list)
for meta_iter in range(meta_iters):
    print '\n\nmeta_iter=%d' % meta_iter
    
    net = Net().double().cuda()
    opt, hist = do_meta_iter(meta_iter, net, lrs.exp(), logit(mos))
    
    lrs, mos = hyperopt.step_w_grads([
        opt.d_lrs * d_exp(lrs),
        opt.d_mos * d_logit(mos)
    ])
    
    all_hist['val'].append(hist['val'])


for l in to_numpy(lrs[:,::2]).T:
    _ = plt.plot(np.exp(l))

show_plot()


_ = plt.plot(np.hstack(all_hist['val']))
_ = plt.ylim(0.9, 1.0)
show_plot()


# # Save results
# f = h5py.File('hist-dev.h5')
# f['train'] = np.vstack(all_hist['train'])
# f['test'] = np.vstack(all_hist['val'])
# f.close()


