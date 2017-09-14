import sys
import numpy as np
from functools import partial

sys.path.append('/home/bjohnson/software/autograd/')
sys.path.append('/home/bjohnson/software/hypergrad')
from funkyyak import grad, kylist, Differentiable

def fill_parser(parser, items):
    partial_vects = [np.full(parser[name].size, items[i]) for i, name in enumerate(parser.names)]
    return np.concatenate(partial_vects, axis=0)

def sgd_parsed(L_grad, hypers, parser, callback=None, forward_pass_only=True):
    x0, alphas, betas, meta = hypers
    X, V = ExactRep(x0), ExactRep(np.zeros(x0.size))
    iters = zip(range(len(alphas)), alphas, betas)
    
    res = []
    for i, alpha, beta in iters:
        g = L_grad(X.val, meta, i)
        res.append(g)
        if callback:
            callback(X.val, V.val, g, i)
        
        cur_alpha_vect = fill_parser(parser, alpha)
        cur_beta_vect  = fill_parser(parser, beta)
        V.mul(cur_beta_vect).sub(g)
        X.add(cur_alpha_vect * V.val)
    
    x_final = X.val
    
    if forward_pass_only:
        return x_final, X.aux
    
    # Hypergradient calculation
    def hypergrad(outgrad):
        d_x = outgrad
        d_alphas, d_betas = np.zeros(alphas.shape), np.zeros(betas.shape)
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        
        grad_proj  = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x    = grad(grad_proj, 0)
        L_hvp_meta = grad(grad_proj, 1)
        
        for i, alpha, beta in iters[::-1]:
            
            # build alpha and beta vector
            cur_alpha_vect = fill_parser(parser, alpha)
            cur_beta_vect  = fill_parser(parser, beta)
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_alphas[i,j] = np.dot(d_x[ixs], V.val[ixs])
            
            # Exactly reverse SGD
            X.sub(cur_alpha_vect * V.val)
            g = L_grad(X.val, meta, i)
            V.add(g).div(cur_beta_vect)
            
            d_v += d_x * cur_alpha_vect
            
            for j, (_, (ixs, _)) in enumerate(parser.idxs_and_shapes.iteritems()):
                d_betas[i,j] = np.dot(d_v[ixs], V.val[ixs])
                
            d_x    -= L_hvp_x(X.val, meta, d_v, i)
            d_meta -= L_hvp_meta(X.val, meta, d_v, i)
            d_v    *= cur_beta_vect
        
        assert np.all(ExactRep(x0).val == X.val)
        return d_x, d_alphas, d_betas, d_meta
        
    return x_final, [None, hypergrad]


sgd_parsed = Differentiable(sgd_parsed, partial(sgd_parsed, forward_pass_only=False))
