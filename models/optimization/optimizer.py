from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
import math

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().lerp_(grad, 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.lerp_(grad, 1 - beta2)
    
    
def get_weight_decay(independent_weight_decay, learning_rate):
    return math.exp(math.log(independent_weight_decay) - math.log(learning_rate))

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, independent_weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    # Parameterize weight decay as independent of learning rate
    weight_decay = get_weight_decay(independent_weight_decay, lr)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas, eps=eps)
