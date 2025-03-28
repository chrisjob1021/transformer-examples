class Config:
    def __init__(self, d_model=256, num_heads=4, max_len=1024, dropout=0.1, rope=False):
        self.d_model = d_model
        self.num_heads = num_heads
        self.per_head_dim = d_model // num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.rope = rope

class MLAConfig(Config):
    def __init__(self, d_model=256, num_heads=4, max_len=1024, dropout=0.1, rope=False):
        super().__init__(d_model, num_heads, max_len, dropout, rope)
        self.d_model_prime_compressed = d_model*num_heads // 16
        self.d_model_compressed = d_model*num_heads // 16