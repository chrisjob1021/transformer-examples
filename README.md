# Transformer Examples

This repository contains a collection of toy implementations and examples of key components from modern Transformer architectures. Each example is designed to be educational, well-documented, and easy to understand.

## Components

| Component | Description | Paper |
|-----------|-------------|-------|
| [Multi-Head Latent Attention (MLA)](./attention/mla_attention.ipynb) | A novel attention mechanism from DeepSeek V2 that uses latent queries to reduce KV cache and Rotary Position Embeddings | [DeepSeek V2 Technical Report](https://arxiv.org/abs/2405.04434) |
| [Multi-Head Attention](./attention/multi_head_attention.py) | The original attention mechanism from the Transformer paper | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [Relative Multi-Head Attention](./attention) | Attention with relative position representations | [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) |
| [Absolute Positional Encoding](./attention/positional_encoding.ipynb) | Sinusoidal positional encoding from the original Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [Rotary Position Embedding](./attention/mla_attention.py) | Enhanced positional encoding using rotation | [RoFormer](https://arxiv.org/abs/2104.09864) |

## Setup

1. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

Each component has its own directory with:
- Implementation code
- Jupyter notebook with examples (and visualizations)

To run a notebook:

1. Make sure Jupyter is installed:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. In your browser, navigate to the component you want to explore (e.g., `attention/mla_attention.ipynb`)
4. Click on the notebook to open it
5. You can run cells individually by pressing `Shift+Enter` or run all cells from the `Cell` menu

For example, to explore Multi-Head Latent Attention from DeepSeek:
```bash
cd attention
jupyter notebook mla_attention.ipynb
```

## Contributing

Contributions are welcome! If you'd like to add a new component or improve an existing one, please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 