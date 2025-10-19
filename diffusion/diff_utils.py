import torch

@torch.cuda.amp.autocast(enabled=False)
def predict_start_from_noise( z_t, noise, alpha2):
    x0 = (z_t - (1-alpha2).sqrt() * noise) / alpha2.sqrt().clamp(min = 1e-8)
    return x0

@torch.cuda.amp.autocast(enabled=False)       
def predict_noise_from_start(z_t, x0, alpha2):
    eps = (z_t - alpha2.sqrt() * x0) / (1-alpha2).sqrt().clamp(min = 1e-8)
    return eps

@torch.cuda.amp.autocast(enabled=False)
def predict_start_from_v(z_t, v, alpha2):
    x0 = alpha2.sqrt() * z_t - (1-alpha2).sqrt() * v
    return x0

@torch.cuda.amp.autocast(enabled=False)
def predict_noise_from_v(z_t, v, alpha2):
    eps = (1-alpha2).sqrt() * z_t + alpha2.sqrt() * v
    return eps

@torch.cuda.amp.autocast(enabled=False)
def predict_v_from_start_and_eps(x, noise, alpha2):
    v = alpha2.sqrt() * noise - x* (1-alpha2).sqrt()
    return v

@torch.cuda.amp.autocast(enabled=False)
def sigmoid_k_weighting(k, gamma):
    return torch.sigmoid(-gamma + k)