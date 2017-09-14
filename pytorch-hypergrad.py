
import sys
sys.path.append('../../../')
import itertools as it
import numpy as np
from copy import copy
from hypergrad.nn_utils import BatchList, VectorParser

import torch
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

np.random.seed(456)

N_weights = 6
num_epochs = 100

parser = VectorParser()
parser.add_shape('first',  [6,])
parser.add_shape('second', [1,])
# parser.add_shape('third',  [3,])
N_weight_types = 1

# --
# Hypergrad

W0     = 0.25 * np.ones(N_weights)
alphas = 0.1 + np.zeros((num_epochs, N_weight_types))
betas  = 0.9 + np.zeros((num_epochs, N_weight_types))
meta   = 1 + np.zeros(N_weights)
target = np.random.uniform(0, 1, W0.shape)

def loss_fun(W, meta, i=None):
    return np.mean((W - target) ** 2)

def hyperloss(params):
    (W0, alphas, betas, meta) = params
    result = sgd_parsed(grad(loss_fun), kylist(W0, alphas, betas, meta), parser)
    return loss_fun(result, meta)

L_grad = grad(loss_fun)
ag_result, ag_aux = sgd_parsed(L_grad, kylist(W0, alphas, betas, meta), parser)


# Exact pytorch SGD

X = ETensor(torch.DoubleTensor(W0))

target_ = Parameter(torch.DoubleTensor(target))
def loss_fun_(W):
    return ((W - target_) ** 2).mean()

V = None
lr = torch.DoubleTensor([0.1])
momentum = torch.DoubleTensor([0.9])
hist = []
for i in range(num_epochs):
    Xpar = Parameter(X.val)
    g = autograd.grad(loss_fun_(Xpar), Xpar, only_inputs=True)[0].data
    
    if not V:
        V = ETensor(g.clone().zero_())
    
    _ = V.mul(momentum)
    _ = V.sub(g)
    _ = X.add(lr * V.val)

torch_result = X.val.numpy()
assert np.allclose(torch_result, ag_result)

# =============================================

# --
# Hyperparameter optimization in `hypergrad`

ag_hgrads = grad(hyperloss)([W0, alphas, betas, meta])
ag_hgrads[1]

# --
# Hyperparameter optimization in torch

iters = zip(range(len(alphas)), alphas, betas)

# Gradient w.r.t. model weights
Xpar = Parameter(X.val)
d_x = autograd.grad(loss_fun_(Xpar), Xpar)[0].data

# Buffers for holding lr + momentum
d_alphas = torch.zeros(alphas.shape).double()
d_betas  = torch.zeros(betas.shape).double()

# Buffer for holding momentum gradients
d_v = torch.zeros(d_x.size()).double()

grad_proj_x = lambda x, d: (autograd.grad(loss_fun_(x), x, create_graph=True)[0] * d).sum()

for i, alpha, beta in iters[::-1]:
    cur_alpha_vect = torch.DoubleTensor(fill_parser(parser, alpha))
    cur_beta_vect  = torch.DoubleTensor(fill_parser(parser, beta))
    
    # Update learning rates
    # for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
    #     d_alphas[i,j] = (d_x[ixs] * V.val[ixs]).sum()
    
    d_alphas[i] = (d_x * V.val).sum()
        
    # Exactly reverse SGD
    _ = X.sub(lr * V.val)
    Xpar = Parameter(X.val)
    g = autograd.grad(loss_fun_(Xpar), Xpar)[0].data
    _ = V.add(g).div(momentum)
    
    d_v += d_x * cur_alpha_vect
    
    # Update momentum
    for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
        d_betas[i,j] = (d_v[ixs] * V.val[ixs]).sum()
    
    # Update weights
    d_vpar = Parameter(d_v)
    d_x -= autograd.grad(grad_proj_x(Xpar, d_vpar), Xpar)[0].data
    
    d_v = d_v * cur_beta_vect

print X.val.numpy()
print V.val.numpy()

assert np.all(ag_hgrads[1] == d_alphas.numpy())