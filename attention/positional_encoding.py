import torch
import torch.nn as nn
import math
from utils import Config

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        self.config = config
        self.past_seq_len = 0

    def forward(self, x):
        with torch.no_grad():
            x = apply_rope(x, self.past_seq_len, visualize=True)
            self.past_seq_len = x.size(2)
        return x

class AbsolutePositionalEncoding(nn.Module):
    """
    Implementation of Absolute Positional Encoding based on the "Attention Is All You Need" paper:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
    Attention Is All You Need. arXiv:1706.03762

    The sinusoidal positional encoding is defined as:
    pe(pos, 2i)   = sin(pos / 10000^(2i/dim))
    pe(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    """

    def __init__(self, config: Config):
        super().__init__()

        self.max_len = config.max_len
        self.d_model = config.d_model
        self.dropout = nn.Dropout(config.dropout)

        pe = torch.zeros(self.max_len, self.d_model)
        pos_within_vector = torch.arange(0, self.max_len, dtype=float).unsqueeze(1)
        every_other_dim = torch.arange(0, self.d_model, 2, dtype=float)
        '''
        a = 2i/dim
        -ln(10000^a) = -a ln(10000) 
        exp(ln(10000^-a)) = 10000^(-a) 
        '''
        div_term = torch.exp((math.log(10000.0)) * -every_other_dim / self.d_model)
        pe[:, 0::2] = torch.sin(pos_within_vector * div_term)
        pe[:, 1::2] = torch.cos(pos_within_vector * div_term)
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
