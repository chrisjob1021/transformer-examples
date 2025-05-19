"""
Implementation of RoFormer (Rotary Transformer) based on the paper:
Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). 
RoFormer: Enhanced Transformer with Rotary Position Embedding. 
arXiv:2104.09864
"""

import attention
import torch.nn as nn
from utils import Config
import torch
import torch.nn.functional as F
import debugpy

class RoFormerDecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.self_attn = attention.MultiHeadAttention(config, self.W_O)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
        # Applies Layer Normalization over a mini-batch of inputs.
        # This layer implements the operation as described in the paper Layer Normalization
        # (Ba et al., 2016: https://arxiv.org/abs/1607.06450)
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)

        # Position-wise feed-forward
        # Using nn.sequential feels like cheating :)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, attention_mask=None):
        # Check if tensors are on CPU
        if x.device.type == 'cpu':
            raise RuntimeError("Input tensor 'x' is on CPU but should be on GPU")
        
        # batch_size, seq_len = x.shape[0], x.shape[1]

        attn_output = self.self_attn(x, attention_mask=attention_mask)
        # attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)

        if attn_output.device.type == 'cpu':
            raise RuntimeError("Attention output tensor is on CPU but should be on GPU")

        x = x + self.dropout1(attn_output)
        x = self.ln1(x)

        ffn_output = self.ffn(x)
        if ffn_output.device.type == 'cpu':
            raise RuntimeError("FFN output tensor is on CPU but should be on GPU")
            
        x = x + self.dropout2(ffn_output)
        x = self.ln2(x)
        return x

class RoFormerDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([RoFormerDecoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.device.type == 'cpu':
            raise RuntimeError("Input tensor 'input_ids' is on CPU but should be on GPU")
        if attention_mask is not None and attention_mask.device.type == 'cpu':
            raise RuntimeError("Attention mask tensor is on CPU but should be on GPU")
            
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)
            
        return x

import os
import json
from typing import Union

class RoFormerForCausalLM(nn.Module):
    def _tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.weight

    def __init__(self, backbone: RoFormerDecoder, config: Config):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._tie_weights()

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, os.PathLike],
        map_location: str = "cpu",
        strict: bool = True,
        dtype: torch.dtype = None,
        **override_config,          # let caller tweak e.g. dropout=0
    ):
        """
        Args
        ----
        path (str): directory containing `config.json` and `pytorch_model.bin`
        map_location: passed to `torch.load`
        strict: forward to `load_state_dict`
        dtype: if set, casts the loaded state‑dict (`torch.float16`, `bfloat16`, …)
        override_config: key‑value pairs to overwrite fields in the JSON config
        """
        # ── 1. load and merge config ─────────────────────────────
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict.update(override_config)               # user overrides
        config = Config(**cfg_dict)

        # ── 2. instantiate model skeleton ────────────────────────
        backbone = RoFormerDecoder(config)
        model = cls(backbone, config)
        
        # ── 3. load weights ──────────────────────────────────────
        # Find all checkpoint files in the directory
        checkpoint_dirs = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoint files found in {path}")
        
        # Sort by checkpoint number to get the highest checkpoint
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint_path = os.path.join(path, latest_checkpoint, "pytorch_model.bin")
        
        state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
        )
        if dtype is not None:
            # cast every tensor to desired dtype *before* loading
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        model._tie_weights()

        # Check if the model is tied
        if strict and (missing or unexpected):
            raise RuntimeError(f"load_state_dict() missing={missing} unexpected={unexpected}")

        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.device.type == 'cpu':
            raise RuntimeError("Input tensor 'input_ids' is on CPU but should be on GPU")
        if attention_mask is not None and attention_mask.device.type == 'cpu':
            raise RuntimeError("Attention mask tensor is on CPU but should be on GPU")
        if labels is not None and labels.device.type == 'cpu':
            raise RuntimeError("Labels tensor is on CPU but should be on GPU")
            
        hidden = self.backbone(input_ids)
        if hidden.device.type == 'cpu':
            raise RuntimeError("Hidden state tensor is on CPU but should be on GPU")
            
        logits = self.lm_head(hidden) # [batch_size, sequence_length, vocab_size]
        if logits.device.type == 'cpu':
            raise RuntimeError("Logits tensor is on CPU but should be on GPU")
            
        loss = None

        if labels is not None:
            # Flatten the logits and labels for cross entropy loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100)
            # This flattening is necessary because PyTorch's cross_entropy expects:
            # Input (logits): [N, C] where N is number of samples and C is number of classes
            # Target (labels): [N] where each value is the correct class index

        if loss is not None and loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return {"loss": loss, "logits": logits}
