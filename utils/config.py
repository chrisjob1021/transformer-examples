class Config:
    def __init__(self, dim=256, num_heads=4, max_len=1024, dropout=0.1):
        self.dim = dim
        self.num_heads = num_heads
        self.per_head_dim = dim // num_heads
        self.max_len = max_len
        self.dropout = dropout

class MLAConfig:
    def __init__(self, dim=256, num_heads=4):
        self.dim = dim
        self.num_heads = num_heads
        self.per_head_dim = dim // num_heads
        self.dim_prime_compressed = self.per_head_dim*num_heads // 16
        self.dim_compressed = self.per_head_dim*num_heads // 16