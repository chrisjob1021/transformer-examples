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
import os
import json
from typing import Union

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

class RoFormerForCausalLM(nn.Module):
    def _tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.weight

    def __init__(self, backbone: RoFormerDecoder, config: Config):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._tie_weights()

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save the model and its configuration to a directory.
        
        Args:
            save_directory (str or os.PathLike): Directory to save the model to.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model's state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration by dynamically getting all non-private attributes
        config_dict = {k: getattr(self.backbone.config, k) 
                      for k in vars(self.backbone.config) 
                      if not k.startswith('_') and not callable(getattr(self.backbone.config, k))}
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Model saved to {save_directory}")

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
        path (str): directory containing `config.json` and `pytorch_model.bin` or HuggingFace model ID
        map_location: passed to `torch.load`
        strict: forward to `load_state_dict`
        dtype: if set, casts the loaded state‑dict (`torch.float16`, `bfloat16`, …)
        override_config: key‑value pairs to overwrite fields in the JSON config
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        # Check if path is a HuggingFace model ID
        if "/" in str(path) and not os.path.exists(path):
            try:
                # Download config and model files from HuggingFace
                config_path = hf_hub_download(repo_id=path, filename="config.json")
                model_path = hf_hub_download(repo_id=path, filename="pytorch_model.bin")
            except Exception as e:
                raise RuntimeError(f"Failed to download model from HuggingFace: {e}")
        else:
            # Local path handling
            config_path = os.path.join(path, "config.json")
            model_path = os.path.join(path, "pytorch_model.bin")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
        # ── 1. load and merge config ─────────────────────────────
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict.update(override_config)               # user overrides
        config = Config(**cfg_dict)

        print(f"Loaded config: {cfg_dict}")

        # ── 2. instantiate model skeleton ────────────────────────
        backbone = RoFormerDecoder(config)
        model = cls(backbone, config)
        
        # ── 3. load weights ──────────────────────────────────────
        state_dict = torch.load(
            model_path,
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
