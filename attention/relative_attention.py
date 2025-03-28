"""
Implementation of Relative Multi-Head Self-Attention based on:
1. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations.
   arXiv:1803.02155

2. Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
   arXiv:2104.09864
"""

import torch
import torch.nn as nn
import math
from utils import Config

class RelativeMultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.d_model = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.max_len = config.max_len

        # Projections for the usual Q, K, V
        self.query_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj   = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj   = nn.Linear(self.d_model, self.d_model)

        self.relative_key_embeddings = nn.Embedding(2 * self.max_len + 1, self.head_dim)
        self.relative_value_embeddings = nn.Embedding(2 * self.max_len + 1, self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Content-based (standard) attention score ---
        # qk shape: (batch_size, num_heads, seq_len, seq_len)
        qk = torch.matmul(q, k.transpose(-2, -1))

        # --- Incorporate relative position (Key) ---
        # Build a matrix of relative position offsets for each pair (i, j)
        pos_ids = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len)

        # If sequence is seq_len, then pos_ids is in range [-(seq_len), (seq_len)]
        # Shifting by max_len puts the range in [0, 2 * max_len + 1]
        # which is the range of the relative key embeddings
        pos_ids = pos_ids + (self.max_len)
        pos_ids = pos_ids.clamp(0, 2*self.max_len + 1)                      # shape (seq_len, seq_len)

        rel_k = self.relative_key_embeddings(pos_ids)                       # shape (seq_len, seq_len, head_dim)
        q_r = q.unsqueeze(3)                                                # shape (batch_size, num_heads, seq_len, 1, head_dim)
        rel_k = rel_k.unsqueeze(0).unsqueeze(0)                             # shape (1, 1, seq_len, seq_len, head_dim)
        qk_r = torch.matmul(q_r, rel_k.transpose(-2, -1)).squeeze(-2)       # shape (batch_size, num_heads, seq_len, seq_len)

        qk = qk + qk_r
        qk_r = qk / math.sqrt(self.head_dim)
        probs = torch.softmax(qk, dim=-1)                                  # shape (batch_size, num_heads, seq_len, seq_len)

        attn = torch.matmul(probs, v) # scores torch.Size([2, 4, 12, 4])          

        # attn = torch.matmul(scores, v) # scores torch.Size([2, 4, 12, 4])          
        # print("attn", attn.shape)                          

        rel_v = self.relative_value_embeddings(pos_ids)                     # shape (seq_len, seq_len, head_dim)

        # Typically we can do a matrix multiplication of the attention probabilities and the values
        # For relative embeddings, we have a separate embedding for each possible distance, per head
        # So, instead of a single matrix with size (seq_len, head_dim), we have a matrix with size (seq_len, seq_len, head_dim)
        # α(i,j) * aV(i,j)
        # Step 1 - element-wise multiply each attention weight by the corresponding relative embedding
        probs_r = probs.unsqueeze(-1)                                       # shape (batch_size, num_heads, seq_len, seq_len, 1)
        rel_v = rel_v.unsqueeze(0).unsqueeze(0)                             # shape (1, 1, seq_len, seq_len, dim_head)
        attn_r = probs_r * rel_v                                            # shape (batch_size, num_heads, seq_len, seq_len, dim_head)
        # Step 2 - sum over the dimension j
        attn_r = attn_r.sum(dim=3)                                           # shape (batch_size, num_heads, seq_len, dim_head)
        
        attn = attn + attn_r
        # Reshape to the original dimensions and consolidate all of the heads
        out = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        # And linearly project
        out = self.out_proj(out)
        return out