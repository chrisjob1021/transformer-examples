from typing import Optional

class Config:
    """Configuration class for transformer model architecture.
    
    Attributes:
        d_model (int): The dimension of the model's hidden states
        num_heads (int): Number of attention heads
        per_head_dim (int): Dimension per attention head
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout rate
        rope (bool): Whether to use RoPE (Rotary Position Embedding)
        ffn_dim (int): Feed-forward network dimension
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope: bool = False,
        num_layers: int = 6,
        per_head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.per_head_dim = per_head_dim if per_head_dim is not None else d_model // num_heads
        self.dropout = dropout
        self.rope = rope
        self.ffn_dim = ffn_dim if ffn_dim is not None else d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

class TrainingConfig:
    """Configuration class for training parameters.
    
    Attributes:
        steps (int): Number of training steps
        batch_size (int): Size of training batches
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
    """
    
    def __init__(
        self,
        steps: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        block_size: int = 1024,
    ) -> None:
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.block_size = block_size

class MLAConfig(Config):
    """Configuration class for MLA (Multi-Linear Attention) model.
    
    Extends the base Config class with additional parameters for MLA-specific features.
    
    Attributes:
        d_model_prime_compressed (int): Compressed dimension for prime features
        d_model_compressed (int): Compressed model dimension
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope: bool = False,
        num_layers: int = 6,
        per_head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope=rope,
            num_layers=num_layers,
            per_head_dim=per_head_dim,
            ffn_dim=ffn_dim
        )
        self.d_model_prime_compressed = d_model * num_heads // 16
        self.d_model_compressed = d_model * num_heads // 16
