
import numpy as np
import torch
from torch import nn
from einops import rearrange
import matplotlib.pyplot as plt

class LossEMASampler(nn.Module):
    def __init__(self, n_bins=100, ema_decay=0.9, gamma_min=-15, gamma_max=15):
        super().__init__()

        self.n_bins = n_bins
        self.ema_decay = ema_decay
        # Register loss bins as a buffer so that it is saved with the model
        self.register_buffer("_loss_bins", torch.ones((n_bins), dtype=torch.float64))
        self.register_buffer("_unweighted_loss_bins", torch.ones((n_bins), dtype=torch.float64))
        gamma_range = gamma_max - gamma_min
        self.bin_length = gamma_range/n_bins
        # Register step as a buffer so that it is saved with the model
        self.register_buffer("step", torch.tensor(0))
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max


    def weights(self, ):
        weights = self._loss_bins.clone()
        return weights

    def update_with_all_losses(self, gamma, losses):
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min

            bin_mask = (gamma >= gamma0) & (gamma < gamma1)
            if bin_mask.any():
                self._loss_bins[i] = self.ema_decay * self._loss_bins[i] + (1-self.ema_decay) * losses[bin_mask].mean().item()
        self.step+=1

    def update_with_all_unweighted_losses(self, gamma, losses):
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min

            bin_mask = (gamma >= gamma0) & (gamma < gamma1)
            if bin_mask.any():
                self._unweighted_loss_bins[i] = self.ema_decay * self._unweighted_loss_bins[i] + (1-self.ema_decay) * losses[bin_mask].mean().item()
        self.step+=1

    def sample(self, batch_size, device, uniform=False):
        if uniform:
            gamma = torch.rand((batch_size), device=device) * (self.gamma_max - self.gamma_min) + self.gamma_min
            density = torch.ones((batch_size), device=device)
            return gamma, density
        else:
            bin_weights = self.weights().to(device)
        bins = torch.multinomial(bin_weights, batch_size, replacement=True).to(device)
        samples = torch.rand((batch_size), device=device) * self.bin_length
        gamma = (samples + bins * self.bin_length + self.gamma_min)
        # Check all samples in [-gamma_min, gamma_max]
        assert (gamma >= self.gamma_min).all() and (gamma <= self.gamma_max).all()
        density = bin_weights[bins]
        return gamma, density
        
    def get_loss_emas(self):
        # Return dict of loss emas where the key is the gamma range
        loss_emas = {}
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._loss_bins[i].item()
        return loss_emas
    
    def get_unweighted_loss_emas(self):
        # Return dict of unweighted loss emas where the key is the gamma range
        loss_emas = {}
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._unweighted_loss_bins[i].item()
        return loss_emas

    def get_normalized_loss_emas(self):
        # Return dict of loss densities where the key is the gamma range
        loss_emas = {}
        denominator = self._loss_bins.sum().item()
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            loss_emas[(gamma0,gamma1)] = self._loss_bins[i].item() / denominator
        return loss_emas

    def get_cdf(self):
        # Return dict of loss cdfs where the key is the gamma range
        cdf_dict = {}
        weights = self.weights().cpu().numpy()
        weights = weights/weights.sum()
        cdf = np.cumsum(weights)
        for i in range(self.n_bins):
            gamma0 = i*self.bin_length + self.gamma_min
            gamma1 = (i+1)*self.bin_length + self.gamma_min
            cdf_dict[(gamma0,gamma1)] = cdf[i].item()
        return cdf_dict
