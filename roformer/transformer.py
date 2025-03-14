import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    A simple Transformer encoder block.
    """
    def __init__(self, d_model, num_heads, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
class Transformer(nn.Module):
    """
    A simple Transformer model, as in Vaswani et al. (2017).
    """
    def __init__(self, d_model, num_heads, num_layers, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder()

        