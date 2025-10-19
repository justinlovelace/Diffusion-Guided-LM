import torch
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


# Avoid log(0)
def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

# noise schedules

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# converting gamma to alpha, sigma or logsnr
@torch.cuda.amp.autocast(enabled=False)
def log_snr_to_alpha2(log_snr):
    alpha2 = torch.sigmoid(log_snr)
    return alpha2

# Log-SNR shifting (https://arxiv.org/abs/2301.10972)
@torch.cuda.amp.autocast(enabled=False)
def alpha2_to_shifted_log_snr(alpha2, scale = 1):
    return (log(alpha2) - log(1 - alpha2)).clamp(min=-20, max=20) + 2*np.log(scale).item()

@torch.cuda.amp.autocast(enabled=False)
def time_to_alpha2(t, alpha2_schedule, scale):
    alpha2 = alpha2_schedule(t)
    shifted_log_snr = alpha2_to_shifted_log_snr(alpha2, scale = scale)
    return log_snr_to_alpha2(shifted_log_snr)

def get_noise_schedule(name):
    if name == 'cosine':
        return partial(cosine_schedule)
    else:
        raise ValueError(f'Unknown noise schedule {name}')
    
def get_scaled_noise_schedule(name, scale):
    unscaled_noise_schedule = get_noise_schedule(name)
    return partial(time_to_alpha2, alpha2_schedule=unscaled_noise_schedule, scale=scale)
