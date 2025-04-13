class Config:
    def __init__(self, d_model=256, num_heads=4, max_len=1024, dropout=0.1, rope=False, ffn_dim=256):
        self.d_model = d_model
        self.num_heads = num_heads
        self.per_head_dim = d_model // num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.rope = rope
        self.ffn_dim = ffn_dim

class TrainingConfig:
    def __init__(self, steps=100, batch_size=32, learning_rate=1e-4, weight_decay=0.01):
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

class MLAConfig(Config):
    def __init__(self, d_model=256, num_heads=4, max_len=1024, dropout=0.1, rope=False, ffn_dim=256):
        super().__init__(d_model, num_heads, max_len, dropout, rope, ffn_dim)
        self.d_model_prime_compressed = d_model*num_heads // 16
        self.d_model_compressed = d_model*num_heads // 16