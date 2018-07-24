#!/usr/bin/env python

"""
    helpers.py
"""

import random
import numpy as np

import torch

def to_numpy(x):
    if type(x) in [list, tuple]:
        return [to_numpy(xx) for xx in x]
    elif type(x) in [np.ndarray, float, int]:
        return x
    elif x.requires_grad:
        return to_numpy(x.detach())
    else:
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

def set_seeds(seed=100):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 123)
    _ = torch.cuda.manual_seed(seed + 456)
    _ = random.seed(seed + 789)


def deterministic_batch(X, y, batch_size, seed):
    idxs = np.random.RandomState(seed).randint(X.size(0), size=batch_size)
    idxs = torch.LongTensor(idxs).cuda()
    return X[idxs], y[idxs]
