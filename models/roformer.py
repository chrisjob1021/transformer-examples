import attention
import torch.nn as nn
from utils import Config
import torch
import torch.nn.functional as F

class RoFormerEncoderLayer(nn.Module):
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

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x_reshaped = x.view(batch_size, seq_len, self.config.num_heads, self.config.per_head_dim).transpose(1, 2)

        attn_output = self.self_attn(q=x_reshaped, k=x_reshaped, v=x_reshaped)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)

        x = x + self.dropout1(attn_output)
        x = self.ln1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.ln2(x)
        return x

class RoFormerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([RoFormerEncoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
            
        return x

import os
import json
from typing import Union

class RoFormerForCausalLM(nn.Module):
    def __init__(self, backbone: RoFormerEncoder, config: Config):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = backbone.embeddings.weight # tie weights between lm head and embeddings

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
        path (str): directory containing `config.json` and `pytorch_model.bin`
        map_location: passed to `torch.load`
        strict: forward to `load_state_dict`
        dtype: if set, casts the loaded state‑dict (`torch.float16`, `bfloat16`, …)
        override_config: key‑value pairs to overwrite fields in the JSON config
        """
        # ── 1. load and merge config ─────────────────────────────
        with open(os.path.join(path, "config.json")) as f:
            cfg_dict = json.load(f)
        cfg_dict.update(override_config)               # user overrides
        config = Config(**cfg_dict)

        # ── 2. instantiate model skeleton ────────────────────────
        backbone = RoFormerEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        model = cls(backbone, config)

        # Weight‑tying must already be done in __init__()
        # (lm_head.weight <- backbone.embeddings.weight)

        # ── 3. load weights ──────────────────────────────────────
        state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"),
            map_location=map_location,
        )
        if dtype is not None:
            # cast every tensor to desired dtype *before* loading
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"load_state_dict() missing={missing} unexpected={unexpected}")

        return model

    def forward(self, input_ids, labels=None):
        hidden = self.backbone(input_ids)
        logits = self.lm_head(hidden) # [batch_size, sequence_length, vocab_size]
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

        return {"loss": loss, "logits": logits}
