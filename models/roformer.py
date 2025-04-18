import attention
import torch.nn as nn
from utils import Config
import torch

class RoFormerEncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.self_attn = attention.MultiHeadAttention(config, self.W_O)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
        # Applies Layer Normalization over a mini-batch of inputs.
        # This layer implements the operation as described in the paper Layer Normalization
        # (Ba et al., 2016: https://arxiv.org/abs/1607.06450)
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)

        # Position-wise feed-forward
        # Using nn.sequential feels like cheating :)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x_reshaped = x.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)

        attn_output = self.self_attn(q=x_reshaped, k=x_reshaped, v=x_reshaped, mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)

        x = x + self.dropout1(attn_output)
        x = self.ln1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.ln2(x)
        return x

class RoFormerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([RoFormerEncoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, mask=None):
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
            
        return x
