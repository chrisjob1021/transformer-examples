from torch import nn
import torch
import numpy as np

class SingleHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        # RoPE added to the query and key
        self.W_K = nn.Linear(config.per_head_dim + config.per_head_dim, config.per_head_dim)
        self.W_Q = nn.Linear(config.per_head_dim + config.per_head_dim, config.per_head_dim)
        self.W_V = nn.Linear(config.per_head_dim, config.per_head_dim)
    
    def forward(self, q, k, v):
        K = self.W_K(k)
        Q = self.W_Q(q)
        V = self.W_V(v)
        return K, Q, V

class MultiHeadAttention(nn.Module):
    def __init__(self, config, W_O):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([SingleHead(config) for _ in range(config.num_heads)])

        self.W_O = W_O

    def forward(self, q, k, v):
        all_K = []
        all_Q = []
        all_V = []

        for head_idx, head in enumerate(self.heads):
            K_new, Q_new, V_new = head(q[:, head_idx], k[:, head_idx], v[:, head_idx])

            K_new = K_new.unsqueeze(1)
            Q_new = Q_new.unsqueeze(1)
            V_new = V_new.unsqueeze(1)

            all_K.append(K_new)
            all_Q.append(Q_new)
            all_V.append(V_new)
        
        all_K = torch.cat(all_K, dim=1)
        all_Q = torch.cat(all_Q, dim=1)
        all_V = torch.cat(all_V, dim=1)

        K = all_K
        V = all_V
        Q = torch.cat([all_Q], dim=-2)

        QK = Q @ K.transpose(-2, -1)
        # config.per_head_dim + config.per_head_dim is accounting for RoPE
        # dh + dRh, which in our case are both equal to per_head_dim
        QK = QK / np.sqrt(self.config.per_head_dim + self.config.per_head_dim)
        QK = torch.softmax(QK, dim=-1)
        o = QK @ V

        batch_size, n_heads, seq_len, per_head_dim = o.shape
        o = o.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.dim)
        
        u = self.W_O(o)

        return u