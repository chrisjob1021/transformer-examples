class RoFormerEncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadLatentAttention(config, rope=True)
        self.ln1 = nn.LayerNorm(config.d_model)