import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from models.modules.norm import RMSNorm
from models.modules.blocks import FeedForward, Attention, CrossAttention
from models.modules.position import AlibiPositionalBias, DynamicPositionBias, AbsolutePositionalEmbedding

def exists(val):
    return val is not None

class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_head, causal, cross_attend=False, context_dim=None, time_emb_dim=None, ff_dropout=0.0, attn_dropout=0.0, cross_attn_dropout=0.0):
        super().__init__()
        self.cross_attend = cross_attend
        if cross_attend:
            assert exists(context_dim), 'context dimension must be supplied if cross attending'
            self.cross_attn = CrossAttention(dim, context_dim, dim_head=dim_head, dropout=cross_attn_dropout)

        self.attn = Attention(dim, dim_head=dim_head, causal=causal, time_cond_dim=time_emb_dim, dropout=attn_dropout)

        self.ff = FeedForward(dim, time_cond_dim=time_emb_dim, dropout=ff_dropout)

    def forward(self, x, attn_bias, context=None, time_emb=None):
        x = x + self.attn(x, attn_bias, time_emb=time_emb)
        if self.cross_attend:
            assert exists(context)
            x = x + self.cross_attn(x, context)
        x = x + self.ff(x, time_emb=time_emb)

        return x

class TransformerModel(nn.Module):
    def __init__(self, dim, num_layers, causal, dim_head=64, dense=False, cross_attend=False, context_dim=None, time_emb_dim=None, pos_emb='absolute', ff_dropout=0.0, attn_dropout=0.0, cross_attn_dropout=0.0):
        super().__init__()
        assert dim % dim_head == 0, 'dimension must be divisible by the head dimension'

        if cross_attend:
            assert exists(context_dim), 'context dimension must be supplied if cross attending'

        n_heads = dim // dim_head

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, dim_head=dim_head, causal=causal, time_emb_dim=time_emb_dim, cross_attend=cross_attend, context_dim=context_dim, ff_dropout=ff_dropout, attn_dropout=attn_dropout, cross_attn_dropout=cross_attn_dropout)
            for i in range(num_layers)
        ])
        self.output_norm = RMSNorm(dim)        

        if pos_emb == 'absolute':
            self.abs_pos_emb = AbsolutePositionalEmbedding(dim)
            self.rel_pos = None
        elif pos_emb == 'alibi':
            assert causal
            self.rel_pos = AlibiPositionalBias(n_heads, n_heads) 
            self.abs_pos_emb = None
        elif pos_emb == 'dpb':
            self.rel_pos = DynamicPositionBias(dim, heads=n_heads, depth=2)
            self.abs_pos_emb = None
        else:
            raise NotImplementedError

        if dense:
            assert num_layers % 2 == 0, 'number of layers must be divisible by 2 for dense connections'
            self.dense_blocks = nn.ModuleList([nn.Linear(dim*2, dim) for i in range(num_layers // 2)])
        self.dense=dense
        self.num_layers = num_layers

    def forward(self, x, context=None, time_emb=None):
        length = x.shape[1]
        attn_bias = self.rel_pos(length, length) if exists(self.rel_pos) else None
        hiddens = []
        if exists(self.abs_pos_emb):
            x = x + self.abs_pos_emb(x)
        for i in range(self.num_layers):
            if self.dense:
                if i < (self.num_layers // 2):
                    hiddens.append(x)
                else:
                    x = torch.cat((x, hiddens.pop()), dim=-1)
                    x = self.dense_blocks[i - (self.num_layers // 2)](x)
            x = self.blocks[i](x, attn_bias=attn_bias, context=context, time_emb=time_emb)
        x = self.output_norm(x)

        return x