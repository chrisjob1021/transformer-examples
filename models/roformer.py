import attention

class RoFormerEncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.self_attn = attention.MultiHeadAttention(config, rope=True)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
        # Applies Layer Normalization over a mini-batch of inputs.
        # This layer implements the operation as described in the paper Layer Normalization
        # (Ba et al., 2016: https://arxiv.org/abs/1607.06450)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)

        # Position-wise feed-forward
        # Using nn.sequential feels like cheating :)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.dropout)