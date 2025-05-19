# RoFormer Implementation

This directory contains an implementation of the RoFormer (Rotary Transformer) model as described in the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). The implementation includes both the model architecture and training scripts.

## Model Architecture

The RoFormer model enhances the standard Transformer architecture by replacing the traditional positional encodings with Rotary Position Embeddings (RoPE). This approach provides several benefits:
- Better extrapolation to longer sequences
- Improved performance on various NLP tasks
- More stable training dynamics

## Training

### Single GPU Training

For single GPU training, you can use the provided Jupyter notebook:
```bash
jupyter notebook roformer_training_single_gpu.ipynb
```

### Multi-GPU Training with Accelerate

For distributed training across multiple GPUs, use the `accelerate` library with `roformer_train.py`:

1. First, configure accelerate:
```bash
accelerate config
```

2. Launch training with accelerate:
```bash
accelerate launch roformer_train.py
```

To resume training from a checkpoint:
```bash
accelerate launch roformer_train.py --resume
```

### Default Output Directory

The model checkpoints and logs will be saved to `/home/ubuntu/roformer` by default. You can specify a different output directory using the `--output_dir` argument:


### Training Configuration

The training script uses the following default configuration:
- Model: 12 layers, 12 attention heads, 768 hidden dimensions
- Training: AdamW optimizer with cosine learning rate schedule
- Batch size: 4 per device with gradient accumulation steps of 16
- Evaluation: Every 1000 steps with early stopping
- Checkpointing: Every 1000 steps, keeping the last 10 checkpoints

### Hardware Requirements

The training script is optimized for distributed training across multiple GPUs. The default configuration assumes:
- 8x NVIDIA V100 GPUs (32GB VRAM each)
- Total batch size: 128 (16 per GPU × 8 GPUs)
- Gradient accumulation steps: 16
- Effective batch size: 2048 (128 × 16)

You can modify these parameters in `roformer_train.py` to suit your needs.

## Model Usage

After training, you can load and use the model:

```python
from models.roformer import RoFormerForCausalLM
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Example text generation
input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Load the trained model
model = RoFormerForCausalLM.from_pretrained("/path/to/saved/model")

# Generate text
output = model(input_ids)
# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
``` 