# Multi-Head Latent Attention (MLA) Toy Implementation

This repository contains a toy implementation of the Multi-Head Latent Attention (MLA) mechanism from the DeepSeek V2 architecture. Multi-Head Latent Attention (MLA) is an efficient attention mechanism introduced in the DeepSeek V2 architecture that reduces the computational complexity of traditional attention from quadratic to linear.

## Installation

### Requirements

To use this implementation, you'll need the following dependencies:

```bash
pip install -r requirements.txt
```

### Development Setup

For development and debugging:

```bash
pip install debugpy
```

## Usage

### Basic Usage

```python
# Example code for using MLA in your models
from mla_attention import MultiHeadLatentAttention

# Initialize the MLA layer
mla = MultiHeadLatentAttention(
    hidden_size=768,
    num_heads=12,
    num_latents=64
)

# Forward pass
attn_output, latent_kv_cache, kr_cache = model(x)
```

### Interactive Notebook

This repository includes a Jupyter notebook (`mla_attention.ipynb`) that demonstrates the implementation and usage of Multi-Head Latent Attention. 

To run the notebook:

```bash
jupyter notebook mla_attention.ipynb
```

### Debugging with debugpy

For remote debugging with debugpy:

```python
# Add this at the beginning of your script or notebook cell
import debugpy

# Configure debugpy to listen on localhost:5678
debugpy.listen(("localhost", 5678))
print("Debugpy is listening on localhost:5678")
```

Then connect your debugger (e.g., VS Code) to localhost:5678. You can add breakpoints in your IDE or via code:

```python
debugpy.breakpoint()
```

## References

- [DeepSeek V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [Official DeepSeek Repository](https://github.com/deepseek-ai/DeepSeek-V2)