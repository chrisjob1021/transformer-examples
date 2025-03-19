import torch
import torch.nn as nn
import math

class AbsolutePositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding described in "Attention Is All You Need", Vasani et. al. [2017]

    pi()   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, config):
        super().__init__()

        self.max_len = config.max_len
        self.d_model = config.d_model
        self.dropout = nn.Dropout(config.dropout)

        pe = torch.zeros(self.max_len, self.d_model)
        pos_within_vector = torch.arange(0, self.max_len, dtype=float).unsqueeze(1)
        every_other_dim = torch.arange(0, self.d_model, 2, dtype=float)
        '''
        a = 2i/d_model
        -ln(10000^a) = -a ln(10000) 
        exp(ln(10000^-a)) = 10000^(-a) 
        '''
        div_term = torch.exp((math.log(10000.0)) * -every_other_dim / d_model)
        pe[:, 0::2] = torch.sin(pos_within_vector * div_term)
        pe[:, 1::2] = torch.cos(pos_within_vector * div_term)
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
