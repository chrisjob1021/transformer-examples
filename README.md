# Transformer Examples

This repository contains a collection of toy implementations and examples of key components from modern Transformer architectures. Each example is designed to be educational, well-documented, and easy to understand.

## Installation

There are two ways to install this package:

1. **For Development (Editable Install)**
```bash
git clone https://github.com/yourusername/transformer-examples.git
cd transformer-examples
pip install -e .
```

2. **As a Package**
```bash
pip install git+https://github.com/yourusername/transformer-examples.git
```

After installation, you can import components like this:
```python
from models.roformer import RoFormerEncoder, RoFormerForCausalLM
```

## Components

| Component | Description | Paper |
|-----------|-------------|-------|
| [Multi-Head Latent Attention (MLA)](./attention/mla_attention.ipynb) | A novel attention mechanism from DeepSeek V2 that uses latent queries to reduce KV cache and Rotary Position Embeddings | [DeepSeek V2 Technical Report](https://arxiv.org/abs/2405.04434) |
| [Multi-Head Attention](./attention/multi_head_attention.ipynb) | The original attention mechanism from the Transformer paper | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [Absolute Positional Encoding](./attention/positional_encoding.ipynb) | Sinusoidal positional encoding from the original Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [Relative Positional Encoding](./attention/relative_attention.ipynb) | Attention with relative position representations | [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) |
| [Rotary Position Embedding](./utils/rope.ipynb) | Enhanced positional encoding using rotation | [RoFormer](https://arxiv.org/abs/2104.09864) |

## Models

| Model | Description | Paper |
|-------|-------------|-------|
| [RoFormer](./models/roformer/README.md) | A Transformer variant that uses Rotary Position Embeddings (RoPE) for enhanced position encoding | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |

## Setup

To set up the development environment, run the following scripts in order:

1. Update Python to version 3.12:
```bash
scripts/10-update_python.sh
```

2. Create and configure the virtual environment:
```bash
scripts/20-create_venv.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Contributing

Contributions are welcome! If you'd like to add a new component or improve an existing one, please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 