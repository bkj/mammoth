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

np.random.seed(123)

N_weights = 6
num_epochs = 1000

parser = VectorParser()
parser.add_shape('first',  [2,])
parser.add_shape('second', [1,])
parser.add_shape('third',  [3,])
N_weight_types = 3

# --
# Helpers

from torch.optim.optimizer import Optimizer, required
class HSGD(Optimizer):
    # """ stripped down torch default, but w/ signs flipped to match hypergrad """
    def __init__(self, params, lr=required, momentum=required):
        super(HSGD, self).__init__(params, {
            "lr" : lr,
            "momentum" : momentum,
        })
        
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                g = param.grad.data
                
                param_state = self.state[param]
                if 'velocity' not in param_state:
                    param_state['velocity'] = g.clone().zero_()
                
                velocity = param_state['velocity']
                
                velocity.mul_(momentum).sub_(g)     # v = rho * v - g
                param.data.add_(group['lr'] * velocity) # w = w + lr * v


# --
# Hypergrad

W0 = 0.25 * np.ones(N_weights)
alphas = 0.1 + np.zeros((num_epochs, N_weight_types))
betas  = 0.9 + np.zeros((num_epochs, N_weight_types))
meta   = 1 + np.zeros(N_weights)
target = np.random.uniform(0, 1, W0.shape)

def loss_fun(W, meta, i=None):
    return np.sum((W - target) ** 2) + np.sum(meta ** 2)

def hyperloss(params):
    (W0, alphas, betas, meta) = params
    result = sgd_parsed(grad(loss_fun), kylist(W0, alphas, betas, meta), parser)
    return loss_fun(result, meta)

L_grad = grad(loss_fun)
ag_result = sgd_parsed(L_grad, kylist(W0, alphas, betas, meta), parser)



W0_     = Parameter(torch.FloatTensor(W0))
alphas_ = Parameter(torch.FloatTensor(alphas))
betas_  = Parameter(torch.FloatTensor(betas))
meta_   = Parameter(torch.FloatTensor(meta))
target_ = Variable(torch.FloatTensor(target))

def loss_fun_(W, meta, i=None):
    return ((W - target_) ** 2).sum() + (meta ** 2).sum()

opt = HSGD([W0_], lr=0.1, momentum=0.9)

for i in range(num_epochs):
    opt.zero_grad()
    loss = loss_fun_(W0_, meta_)
    loss.backward()
    opt.step()

torch_result = W0_.data.numpy()
assert np.allclose(torch_result, ag_result)
# !! This verifies that the optimizers are working correctly

# --
# Hyperparameter optimization in `hypergrad`
ag_hgrads = grad(hyperloss)([W0, alphas, betas, meta])

# --
# Hyperparameter optimization in torch

iters = zip(range(len(alphas)), alphas, betas)

# Gradient w.r.t. model weights
d_x = autograd.grad(loss_fun_(W0_, meta), W0_, create_graph=True)[0]
# Model weights
X = W0_.clone()
# Finishing velocity
V = Parameter(opt.state.values()[0]['velocity'])

# Buffers for holding lr + momentum
d_alphas = torch.zeros(alphas.shape) 
d_betas = torch.zeros(betas.shape)

# Buffer for holding momentum gradients
d_v = Variable(torch.zeros(d_x.size()))

grad_proj_x = lambda x, meta, d: (autograd.grad(loss_fun_(x, meta), x, create_graph=True)[0] * d).sum()

for i, alpha, beta in iters[::-1]:
    print i
    
    cur_alpha_vect = Variable(torch.FloatTensor(fill_parser(parser, alpha)))
    cur_beta_vect  = Variable(torch.FloatTensor(fill_parser(parser, beta)))
    
    # Update learning rates
    for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
        d_alphas[i,j] = (d_x[ixs] * V[ixs]).sum().data[0]
    
    # Exactly reverse SGD
    X -= cur_alpha_vect * V
    g = autograd.grad(loss_fun_(X, meta_), X, create_graph=True)[0]
    V = (V + g) / cur_beta_vect
    
    d_v += d_x * cur_alpha_vect
    
    # Update momentum
    for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
        d_betas[i,j] = (d_v[ixs] * V[ixs]).sum().data[0]
    
    # Update weights
    d_x -= autograd.grad(grad_proj_x(X, meta_, d_v), X, create_graph=True)[0]
    
    d_v = d_v * cur_beta_vect
    
    # print X.data.numpy()
    # print V.data.numpy()
