"""
Implementation of Multi-Head Latent Attention (MLA) and RoPE based on DeepSeek-V2 paper:
DeepSeek-V2: An Open Source Model with DeepSeek-V1's Performance and 2x Training Speed
https://arxiv.org/abs/2402.19526
"""

from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import debugpy
from attention.multi_head_attention import MultiHeadAttention
from utils import MLAConfig
from utils import apply_rope

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: MLAConfig):
        super().__init__()

        self.config = config

        self.W_DKV = nn.Linear(config.d_model, config.d_model_compressed)
        self.W_UK = nn.Linear(config.d_model_compressed, config.num_heads*config.per_head_dim)
        self.W_UV = nn.Linear(config.d_model_compressed, config.num_heads*config.per_head_dim)
        
        self.W_DQ = nn.Linear(config.d_model, config.d_model_prime_compressed)
        self.W_UQ = nn.Linear(config.d_model_prime_compressed, config.num_heads*config.per_head_dim)
        
        self.W_KR = nn.Linear(config.d_model, config.d_model)
        self.W_QR = nn.Linear(config.d_model, config.d_model)

        self.W_O = nn.Linear(config.d_model, config.d_model)

        self.LatentKVAttention = LatentKVAttention(config, self.W_DKV, self.W_UK, self.W_UV)
        self.LatentQAttention = LatentQAttention(config, self.W_DQ, self.W_UQ)
        self.MultiHeadAttention = MultiHeadAttention(config, self.W_O)
    
    def KR(self, h, kr_cache=None):
        k_R_t_new = self.W_KR(h)

        if kr_cache is None:
            print("No RoPE K cache")
            k_R_t = k_R_t_new
        else:
            print("RoPE K cache", kr_cache.shape)
            k_R_t = torch.cat([kr_cache, k_R_t_new], dim=-2)

        kr_cache = k_R_t

        batch_size, seq_len, dim = k_R_t.shape    
        k_R_t = k_R_t.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)
        print("Visualizing RoPE K vector pairs")
        k_R_t = apply_rope(k_R_t, visualize=True)

        return k_R_t, kr_cache

    def forward(self, h, latent_kv_cache=None, kr_cache=None):
        k_C_t, v_C_t, latent_kv_cache = self.LatentKVAttention(h, latent_kv_cache)
        if kr_cache is not None:
            past_seq_len = kr_cache.shape[1]
        else:
            past_seq_len = 0    

        debugpy.breakpoint()
        k_R_t, kr_cache = self.KR(h, kr_cache)
        batch_size, seq_len, dim = k_C_t.shape
        k_C_t = k_C_t.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)

        k_t = torch.cat([k_C_t, k_R_t], dim=-1)

        batch_size, seq_len, dim = v_C_t.shape
        v_C_t = v_C_t.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)
        q_C_t = self.LatentQAttention(h)

        batch_size, seq_len, dim = q_C_t.shape
        q_C_t = q_C_t.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)

        print("Visualizing RoPE Q vector pairs")
        q_R_t = apply_rope(q_C_t, past_seq_len, visualize=True)

        q_t = torch.cat([q_C_t, q_R_t], dim=-1)

        v_t = v_C_t 

        out = self.MultiHeadAttention(q_t, k_t, v_t)
        return out, latent_kv_cache, kr_cache
        
class LatentKVAttention(nn.Module):
    def __init__(self, config, W_DKV, W_UK, W_UV):
        super().__init__()

        self.W_DKV = W_DKV
        self.W_UK = W_UK
        self.W_UV = W_UV
    
    def forward(self, h, latent_kv_cache=None):
        c_KV_t_new = self.W_DKV(h)

        if latent_kv_cache is None:
            print("No Latent KV cache")
            c_KV_t = c_KV_t_new
        else:
            print("Latent KV cache", latent_kv_cache.shape)
            c_KV_t = torch.cat([latent_kv_cache, c_KV_t_new], dim=-2)

        latent_kv_cache = c_KV_t

        k_C_t = self.W_UK(c_KV_t)
        v_C_t = self.W_UV(c_KV_t)

        return k_C_t, v_C_t, latent_kv_cache

class LatentQAttention(nn.Module):
    def __init__(self, config, W_DQ, W_UQ):
        super().__init__()
        self.W_DQ = W_DQ
        self.W_UQ = W_UQ

    def forward(self, h):
        c_Q_t = self.W_DQ(h)
        q_C_t = self.W_UQ(c_Q_t)
        return q_C_t