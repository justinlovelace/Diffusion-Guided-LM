import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# See https://arxiv.org/abs/2303.00848 for details on the weighting functions
def convert_vweight_to_epsweight(v_weight, gamma, monotonic=False):
    eps_weight = v_weight*(torch.exp(-gamma)+1)
    if monotonic:
        neg_eps_weight = torch.flip(eps_weight, dims=[0])
        monotonic_neg_eps_weight = torch.cummax(neg_eps_weight, dim=0).values
        eps_weight = torch.flip(monotonic_neg_eps_weight, dims=[0])
    denom = eps_weight.max()
    return torch.exp(torch.log(eps_weight) - torch.log(denom))

def convert_epsweight_to_vweight(eps_weight, gamma):
    v_weight = torch.exp(torch.log(eps_weight)-torch.log(torch.exp(-gamma)+1))
    denom = v_weight.max()
    return torch.exp(torch.log(v_weight) - torch.log(denom))

def convert_epsweight_to_xweight(eps_weight, gamma):
    x_weight = eps_weight*(torch.exp(gamma))
    denom = x_weight.max()
    return torch.exp(torch.log(x_weight) - torch.log(denom))

class V_Weighting(nn.Module):
    def __init__(self, gamma_shift = 0.0):
        super().__init__()
        gamma_shift = gamma_shift
        self.min_gamma = -15

        self.gamma_linspace = torch.linspace(-15, 15, 10000)
        v_weights = self.v_weighting(self.gamma_linspace, gamma_shift)
        self.register_buffer('v_weights', v_weights)
        eps_weights = convert_vweight_to_epsweight(self.v_weights, self.gamma_linspace, monotonic=True)
        self.register_buffer('eps_weights', eps_weights)
        self.v_weights = convert_epsweight_to_vweight(self.eps_weights, self.gamma_linspace)

        x_weights = convert_epsweight_to_xweight(self.eps_weights, self.gamma_linspace)
        self.register_buffer('x_weights', x_weights)

    def v_weighting(self, gamma, gamma_shift=0.0):
        gamma = gamma - gamma_shift
        v_weighting = 1/torch.cosh(-(gamma/2))
        return v_weighting
    
    def v_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.v_weights[gamma_idx]
    
    def eps_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.eps_weights[gamma_idx]
    
    def x_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.x_weights[gamma_idx]
            
class LogCauchy_V_Weighting(nn.Module):
    def __init__(self, gamma_mean=0.0, gamma_std=0.0, objective='pred_v'):
        super().__init__()
        self.gamma_mean = gamma_mean
        self.gamma_std = gamma_std
        self.min_gamma = -15
        self.max_gamma = 15

        assert objective == 'pred_v'

        cauchy_dist = torch.distributions.cauchy.Cauchy(self.gamma_mean, self.gamma_std)

        self.gamma_linspace = torch.linspace(self.min_gamma, self.max_gamma, 10000)
        v_weights = torch.exp(cauchy_dist.log_prob(self.gamma_linspace))
        self.register_buffer('v_weights', v_weights)
        eps_weights = convert_vweight_to_epsweight(self.v_weights, self.gamma_linspace, monotonic=True)
        self.register_buffer('eps_weights', eps_weights)
        self.v_weights = convert_epsweight_to_vweight(self.eps_weights, self.gamma_linspace)

        x_weights = convert_epsweight_to_xweight(self.eps_weights, self.gamma_linspace)
        self.register_buffer('x_weights', x_weights)

        
    def v_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.v_weights[gamma_idx]
    
    def eps_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.eps_weights[gamma_idx]

    def x_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.x_weights[gamma_idx]

class LogNormal_V_Weighting(nn.Module):
    def __init__(self, mean, std, monotonic=True):
        super().__init__()
        self.gamma_mean = mean
        self.gamma_std = std
        self.min_gamma = -15
        self.max_gamma = 15
        self.monotonic = monotonic

        normal_dist = torch.distributions.normal.Normal(self.gamma_mean, self.gamma_std)

        gamma_linspace = torch.linspace(self.min_gamma, self.max_gamma, 10000)
        self.register_buffer('gamma_linspace', gamma_linspace)
        v_weights = torch.exp(normal_dist.log_prob(self.gamma_linspace))
        self.register_buffer('v_weights', v_weights)
        eps_weights = convert_vweight_to_epsweight(self.v_weights, self.gamma_linspace, monotonic=self.monotonic)
        self.register_buffer('eps_weights', eps_weights)
        self.v_weights = convert_epsweight_to_vweight(self.eps_weights, self.gamma_linspace)

        x_weights = convert_epsweight_to_xweight(self.eps_weights, self.gamma_linspace)
        self.register_buffer('x_weights', x_weights)

    def v_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.v_weights[gamma_idx]
    
    def eps_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.eps_weights[gamma_idx]
    
    def x_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.x_weights[gamma_idx]
        
    
class Asymmetric_LogNormal_V_Weighting(nn.Module):
    def __init__(self, mean, std, std_mult):
        super().__init__()
        self.gamma_mean = mean
        self.gamma_std = std
        self.min_gamma = -15
        self.max_gamma = 15
        gamma_linspace = torch.linspace(self.min_gamma, self.max_gamma, 10000)
        self.register_buffer('gamma_linspace', gamma_linspace)

        neg_cauchy_v_weighting = torch.distributions.cauchy.Cauchy(self.gamma_mean, self.gamma_std*std_mult)
        normal_v_weighting = torch.distributions.normal.Normal(self.gamma_mean, self.gamma_std)
        v_cauchy_weights = (neg_cauchy_v_weighting.log_prob(self.gamma_linspace))
        v_cauchy_weights = torch.exp(v_cauchy_weights - v_cauchy_weights.max())
        v_normal_weights = (normal_v_weighting.log_prob(self.gamma_linspace))
        v_normal_weights = torch.exp(v_normal_weights - v_normal_weights.max())
        v_weights = torch.where(self.gamma_linspace < self.gamma_mean, v_cauchy_weights, v_normal_weights)
        self.register_buffer('v_weights', v_weights)

        eps_weights = convert_vweight_to_epsweight(self.v_weights, self.gamma_linspace, monotonic=True)
        self.register_buffer('eps_weights', eps_weights)
        self.v_weights = convert_epsweight_to_vweight(self.eps_weights, self.gamma_linspace)

        x_weights = convert_epsweight_to_xweight(self.eps_weights, self.gamma_linspace)
        self.register_buffer('x_weights', x_weights)

    def v_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.v_weights[gamma_idx]
    
    def eps_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.eps_weights[gamma_idx]
    
    def x_loss_weighting(self, gamma):
        # Look up nearest gamma in linspace
        gamma_idx = torch.argmin(torch.abs(gamma.unsqueeze(0) - self.gamma_linspace.unsqueeze(1)), dim=0)
        return self.x_weights[gamma_idx]
    
