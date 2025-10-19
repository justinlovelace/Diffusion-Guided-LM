import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from models.modules.norm import RMSNorm

def exists(val):
    return val is not None

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        time_cond_dim = None,
        glu = True,
        dropout = 0.,
        no_bias = False,
        init_zeros = False,
        hidden_dim = None,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(dim)
        # adjust inner_dim for extra glu projection
        assert not (exists(hidden_dim) and exists(mult)), 'hidden_dim and mult are mutually exclusive'
        if exists(hidden_dim):
            inner_dim = hidden_dim
            assert not glu, 'glu is not compatible with hidden_dim'
        else:
            inner_dim = int(dim * mult*2/3) if glu else int(dim * mult)
        dim_out = dim

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )
            zero_init_(self.time_cond[1])

        activation = nn.SiLU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        if init_zeros:
            zero_init_(self.net[-1])

    def forward(self, x, time_emb = None):
        x = self.pre_norm(x)
        if exists(self.time_cond):
            assert exists(time_emb)
            scale, shift = self.time_cond(time_emb).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        x = self.net(x)

        return x
    
def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            time_cond_dim=None,
            dropout=0.,
            causal=False,
            ):
        super().__init__()
        assert dim % dim_head == 0, 'dimension must be divisible by the head dimension'
        self.heads = dim // dim_head
        self.dropout = dropout
        self.pre_norm = RMSNorm(dim)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim*2),
                Rearrange('b d -> b 1 d'),
            )
            zero_init_(self.time_cond[1])

        self.causal = causal
    
    def forward(self, x, attn_bias, time_emb=None):
        batch, l, d = x.shape

        x = self.pre_norm(x)

        if exists(time_emb):
            scale, shift = self.time_cond(time_emb).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        # handle alibi positional bias
        # convert from bool to float
        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, self.heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            if self.causal:
                q_len, k_len = q.shape[-2], k.shape[-2]
                mask_value = -torch.finfo(q.dtype).max
                causal_mask = create_causal_mask(q_len, k_len, device = q.device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)

        # scaled_dot_product_attention handles attn_mask either as bool or additive bias
        # make it an additive bias here

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out
    
class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            context_dim,
            dim_head=64,
            time_cond_dim=None,
            dropout=0.,
            ):
        super().__init__()
        assert dim % dim_head == 0, 'dimension must be divisible by the head dimension'
        self.heads = dim // dim_head
        self.dropout = dropout
        self.pre_norm = RMSNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim*2),
                Rearrange('b d -> b q d'),
            )
            zero_init_(self.time_cond[1])

    
    def forward(self, x, context, time_emb=None):
        batch, l, d = x.shape

        x = self.pre_norm(x)

        if exists(time_emb):
            scale, shift = self.time_cond(time_emb).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out