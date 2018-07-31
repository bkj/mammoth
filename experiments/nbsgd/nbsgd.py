#!/usr/bin/env python

"""
    nbsgd.py
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

sys.path.append('../../../mammoth')
from mammoth.helpers import to_numpy, set_seeds
from mammoth.hyperlayer import HyperLayer
from mammoth.optim import LambdaAdam

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --
# Helpers

def _calc_nb(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))

def calc_r(y_i, x, y, mode='nb'):
    if mode == 'nb':
        return _calc_nb(y_i, x, y)
    elif mode == 'one':
        return np.ones(x.shape[1]) # * _calc_nb(y_i, x, y).mean()
    else:
        raise Exception('unknown mode=%s' % mode)

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-train', type=str, default='./data/aclImdb/X_train.npy')
    parser.add_argument('--x-train-words', type=str, default='./data/aclImdb/X_train_words.npy')
    parser.add_argument('--y-train', type=str, default='./data/aclImdb/y_train.npy')
    
    parser.add_argument('--x-test', type=str, default='./data/aclImdb/X_test.npy')
    parser.add_argument('--x-test-words', type=str, default='./data/aclImdb/X_test_words.npy')
    parser.add_argument('--y-test', type=str, default='./data/aclImdb/y_test.npy')
    
    parser.add_argument('--meta-iters', type=int, default=20)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr-init', type=float, default=0.5)
    parser.add_argument('--mo-init', type=float, default=-2)
    parser.add_argument('--hyper-lr', type=float, default=0.1)
    parser.add_argument('--fixed-data', action="store_true")
    parser.add_argument('--one-r', action="store_true")
    
    parser.add_argument('--learn-lrs', action="store_true")
    parser.add_argument('--learn-mos', action="store_true")
    parser.add_argument('--learn-meta', action="store_true")
    parser.add_argument('--learn-init', action="store_true")
    parser.add_argument('--untrain', action="store_true")
    
    parser.add_argument('--vocab-size', type=int, default=200000)
    
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    if args.learn_init:
        assert args.untrain, 'args.learn_init and not args.untrain'
    
    return args

args = parse_args()
print(json.dumps(vars(args)), file=sys.stderr)

set_seeds(args.seed)

# --
# IO

X_train       = np.load(args.x_train).item()
X_train_words = np.load(args.x_train_words).item()
y_train       = np.load(args.y_train)

X_test        = np.load(args.x_test).item()
X_test_words  = np.load(args.x_test_words).item()
y_test        = np.load(args.y_test)

X_valid_words, X_test_words, y_valid, y_test = train_test_split(X_test_words, y_test, train_size=0.5)

data = {
    "X_train" : torch.from_numpy(X_train_words.toarray()).long(),
    "y_train" : torch.from_numpy(y_train).long(),
    "X_valid" : torch.from_numpy(X_valid_words.toarray()).long(),
    "y_valid" : torch.from_numpy(y_valid).long(),
    "X_test"  : torch.from_numpy(X_test_words.toarray()).long(),
    "y_test"  : torch.from_numpy(y_test).long(),
}
data = {k:v.cuda() for k,v in data.items()}

# # >>
# with open('X_train.ft', 'w') as f:
#     for x, y in zip(X_train_words.toarray(), y_train):
#         x = ' '.join(x[x != 0].astype(str))
#         x = ' '.join(['__label__%d' % y, x])
#         _ = f.write(x + '\n')

# with open('X_valid.ft', 'w') as f:
#     for x, y in zip(X_valid_words.toarray(), y_valid):
#         x = ' '.join(x[x != 0].astype(str))
#         x = ' '.join(['__label__%d' % y, x])
#         _ = f.write(x + '\n')

# with open('X_test.ft', 'w') as f:
#     for x, y in zip(X_test_words.toarray(), y_test):
#         x = ' '.join(x[x != 0].astype(str))
#         x = ' '.join(['__label__%d' % y, x])
#         _ = f.write(x + '\n')

# # <<


# --
# Model definition

def loss_fn(x, y):
    # Numerically stable way to compute this
    x = 2 * x
    y = y.float()
    return (F.softplus(-x.abs()) + x * ((x > 0).float() - y)).mean()

# def loss_fn(x, y):
#     return F.binary_cross_entropy(F.sigmoid(2 * x), y.float())


class DotProdNB(nn.Module):
    def __init__(self, vocab_size, meta):
        
        super().__init__()
        
        # Init w
        self.w_weight    = torch.zeros(vocab_size + 1).uniform_(-0.1, 0.1)
        self.w_weight[0] = 0
        self.w_weight    = nn.Parameter(self.w_weight)
        
        self.r_weight = meta[2:]
        self.w_adj    = meta[0]
        self.r_adj    = meta[1]
        
    def forward(self, feat_idx):
        n_docs, n_words = feat_idx.shape
        
        feat_idx  = feat_idx.view(-1)
        zero_mask = (feat_idx == 0)
        
        w = (self.w_weight[feat_idx] + self.w_adj)
        w[zero_mask] = 0
        w = w.view(n_docs, n_words)
        
        r = self.r_weight[feat_idx]
        r[zero_mask] = 0
        r = r.view(n_docs, n_words)
        
        x = (w * r).sum(dim=1)
        x =  x / self.r_adj
        
        return x

# --
# Define model

def make_hparams(lr_init, mo_init, num_iters, n_groups, mode='nb'):
    meta = np.hstack([[0.4, 10], [0.0], calc_r(1, X_train, y_train, mode=mode)])
    
    return {
        "lrs"  : torch.FloatTensor(np.full((num_iters, n_groups), lr_init)).cuda(),
        "mos"  : torch.FloatTensor(np.full((num_iters, n_groups), mo_init)).cuda(),
        "meta" : torch.FloatTensor(meta).cuda(),
    }

# --
# Run

hparams = make_hparams(
    lr_init=args.lr_init,
    mo_init=args.mo_init,
    num_iters=args.num_iters,
    n_groups=1,
    mode='nb' if not args.one_r else 'one',
)

hparams['lrs'].requires_grad_(args.learn_lrs)
hparams['mos'].requires_grad_(args.learn_mos)
hparams['meta'].requires_grad_(args.learn_meta)

net = DotProdNB(args.vocab_size, meta=hparams['meta']).cuda()

params = [h for h in hparams.values() if h.requires_grad]
if args.learn_init:
    params += list(net.parameters())

if params:
    hopt = LambdaAdam(
        params=params,
        lr=args.hyper_lr,
    )
else:
    hopt = None
    print('------ no hopt ------', file=sys.stderr)


t = time()
for meta_iter in range(args.meta_iters):
    
    if hopt is not None:
        _ = hopt.zero_grad()
    
    hlayer = HyperLayer(
        net=net, 
        hparams={
            "lrs"  : hparams['lrs'].clamp(min=1e-5, max=10),
            "mos"  : 1 - 10 ** hparams['mos'].clamp(max=0),
            "meta" : hparams['meta'] if args.learn_meta else None,
        },
        params=net.parameters(),
        num_iters=args.num_iters, 
        batch_size=args.batch_size,
        seed=(args.seed + 11111) if args.fixed_data else (args.seed + meta_iter),
        verbose=args.verbose,
        loss_fn=loss_fn,
    )
    train_hist, val_acc, test_acc = hlayer.run(
        data=data,
        learn_lrs=args.learn_lrs,
        learn_mos=args.learn_mos,
        learn_meta=args.learn_meta,
        learn_init=args.learn_init,
        untrain=args.untrain,
    )
    
    if hopt is not None:
        _ = hopt.step()
    
    print(json.dumps({
        "meta_iter" : meta_iter,
        "train_acc" : float(np.mean([t['acc'] for t in train_hist[-10:]])),
        "val_acc"   : val_acc,
        "test_acc"  : test_acc,
        "time"      : time() - t,
    }))
    sys.stdout.flush()
