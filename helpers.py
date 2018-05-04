#!/usr/bin/env python

"""
    helpers.py
"""

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

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available:
        _ = torch.cuda.manual_seed(seed)
