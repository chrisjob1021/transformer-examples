"""
Implementation of Multi-Head Attention based on the original "Attention Is All You Need" paper:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need. arXiv:1706.03762
"""

from torch import nn
import torch
import numpy as np
from attention.positional_encoding import RotaryPositionalEncoding
import debugpy

class MultiHeadAttention(nn.Module):
    def __init__(self, config, W_O):
        super().__init__()
        self.config = config

        # # Input dimension depends on whether RoPE is enabled
        # input_dim = config.per_head_dim * 2 if config.rope else config.per_head_dim
        
        # Adjust linear layer input dimensions based on RoPE setting
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        
        self.rope = None
        if config.rope:
            self.rope = RotaryPositionalEncoding(config)

        self.W_O = W_O

    def forward(self, h, k_proj=None, v_proj=None, multi_input_vector=False, attention_mask=None):
        batch_size, seq_len = h.size(0), h.size(1)

        if multi_input_vector:
            h = q_proj
        else:
            q_proj = self.W_Q(h).view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)
            k_proj = self.W_K(h).view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)
            v_proj = self.W_V(h).view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)

        qk = q_proj @ k_proj.transpose(-2, -1).to(q_proj.device)
        qk = qk / np.sqrt(self.config.per_head_dim)

        if attention_mask is not None:
            # attention_mask shape: [seq_len] of 1's and 0's
            # Convert to proper attention mask matrix where:
            # - if mask[j] = 0, position j should be masked out for all queries
            attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            attention_mask = attention_mask.unsqueeze(0)  # [1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(0)  # [1, 1, 1, seq_len]
            attention_mask = attention_mask.expand(batch_size, self.config.num_heads, seq_len, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            attention_mask = attention_mask.to(qk.device)  # Move to same device as qk
            qk = qk.masked_fill(attention_mask == 0, float("-inf"))

        # Create causal mask
        # First, torch.ones(4, 4) creates a 4x4 matrix of ones:
        # 1 1 1 1
        # 1 1 1 1
        # 1 1 1 1
        # 1 1 1 1

        # Then, torch.triu(..., diagonal=1) creates a mask with ones above the diagonal:
        # 0 1 1 1
        # 0 0 1 1
        # 0 0 0 1
        # 0 0 0 0
        
        # The .bool() converts the mask to a boolean tensor where True indicates a masked position.
        # The .unsqueeze(0).unsqueeze(0) adds two dimensions to the mask, making it a 4D tensor:
        # [[[False True True True],
        #   [False False True True],
        #   [False False False True],
        #   [False False False False]]]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.to(qk.device)  # Move to same device as qk

        # Use causal mask
        # When this mask is used with masked_fill(causal_mask, float("-inf")), 
        # it sets all True values to negative infinity. 
        # In the attention mechanism, this effectively means:
        #   Position 0 can only attend to position 0
        #   Position 1 can attend to positions 0 and 1
        #   Position 2 can attend to positions 0, 1, and 2
        #   Position 3 can attend to positions 0, 1, 2, and 3
        # This creates the causal (autoregressive) property where each token can only attend to itself
        # and previous tokens, which is crucial for tasks like language modeling 
        # where you want to prevent the model from "seeing into the future" during training and inference.
        qk = qk.masked_fill(causal_mask, float("-inf"))
        
        # attn_probs.shape == [batch_size, num_heads, seq_len, seq_len]
        attn_probs = torch.softmax(qk, dim=-1) # e.g. attn_probs[0][0][1] = first batch item, first head, second query position
        
        # v_proj.shape == [batch_size, num_heads, seq_len, per_head_dim]
        attn_scores = attn_probs @ v_proj # e.g. output = attn_probs[0][0][1][0] * v_proj[..., 0, :] + attn_probs[0][0][1][1] * v_proj[..., 1, :]
        
        # later positions are masked out
        # attn_scores.shape == [batch_size, num_heads, seq_len, per_head_dim]
        # e.g. attn_scores[0][0][1]
        # then for each:
        #   batch (which input in the batch)
        #   head (which attention head)
        #   query position (which token is querying)
        #   feature (which dimension of the value vector)

        debugpy.breakpoint()
        o = attn_scores.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        
        u = self.W_O(o)

        return u