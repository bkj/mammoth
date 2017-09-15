from torch.optim.optimizer import Optimizer
class HSGD(Optimizer):
    def __init__(self, params, lrs, momentums, num_iters, cuda=False):
        super(HSGD, self).__init__(params, {
            "lrs" : lrs,
            "momentums" : momentums,
        })
        self.d_lrs       = torch.zeros(num_iters).double()
        self.d_momentums = torch.zeros(num_iters).double()
        
        self.cuda = cuda
        if self.cuda:
            self.d_lrs = self.d_lrs.cuda()
            self.d_momentums = self.d_momentums.cuda()
        
    def step(self, i):
        for group in self.param_groups:
            momentum = torch.DoubleTensor([group['momentums'][i]])
            lr = torch.DoubleTensor([group['lrs'][i]])
            
            if self.cuda:
                momentum = momentum.cuda()
                lr = lr.cuda()
                
            for param in group['params']:
                if param.grad is None:
                    continue
                
                g = param.grad.data
                
                param_state = self.state[param]
                
                if 'X' not in param_state:
                    if self.cuda:
                        param_state['X'] = ETensorCUDA(param.data.clone())
                    else:
                        param_state['X'] = ETensor(param.data.clone())
                    
                if 'V' not in param_state:
                    if self.cuda:
                        param_state['V'] = ETensorCUDA(g.clone().zero_())
                    else:
                        param_state['V'] = ETensor(g.clone().zero_())
                
                _ = param_state['V'].mul(momentum).sub(g)
                _ = param_state['X'].add(lr * param_state['V'].val)
                param.data.set_(param_state['X'].val)
